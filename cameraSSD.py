# -*- coding: utf-8 -*-
"""
Created on Tue Oct 09 10:07:55 2018

@author: fabio.roncato
"""
# import the needed modules
import os
import sys
import numpy as np
import json
import cv2
import pika
import time
import datetime
from transform import CoordinateTransform
from reprojecter import ReprojectTo4326
from pyimagesearch.centroidtracker import CentroidTracker

from mvnc import mvncapi as mvnc

#from shapely.geometry import Point
#from shapely.geometry.polygon import Polygon

dim=(300,300)

LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
# upon successful return this image will be overlayed with boxes
# and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
# be used to peform the inference.
def run_inference(image_to_classify, ssd_mobilenet_graph):

    # the minimal score for a box to be shown
    min_score_percent = 60

    # get a resized version of the image that is the dimensions
    # SSD Mobile net expects
    resized_image = preprocess_image(image_to_classify)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(np.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = ssd_mobilenet_graph.GetResult()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])
    print('total num boxes: ' + str(num_valid_boxes))
    
    # list of the index accepted
    accepted = ["person"]
    
    # the returned object of type in the list "accepted" recognized
    person = []

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not np.isfinite(output[base_index]) or
                    not np.isfinite(output[base_index + 1]) or
                    not np.isfinite(output[base_index + 2]) or
                    not np.isfinite(output[base_index + 3]) or
                    not np.isfinite(output[base_index + 4]) or
                    not np.isfinite(output[base_index + 5]) or
                    not np.isfinite(output[base_index + 6])):
                # boxes with non infinite (inf, nan, etc) numbers must be ignored
                print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
                continue
                       
            # read the score value of the current recognized ID
            percentage = int(output[base_index + 2] * 100)
            
            #verify the type ID is correct and the score is greather the imposed threshold
            if(LABELS[int(output[base_index + 1])] in str(accepted) and
            percentage >= min_score_percent):
                # clip the boxes to the image size incase network returns boxes outside of the image
                x1 = max(0, int(output[base_index + 3] * image_to_classify.shape[1]))
                y1 = max(0, int(output[base_index + 4] * image_to_classify.shape[0]))
                x2 = min(image_to_classify.shape[0], int(output[base_index + 5] * image_to_classify.shape[1]))
                y2 = min(image_to_classify.shape[1], int(output[base_index + 6] * image_to_classify.shape[0]))

                # the position foundin the original frame have to be resized according to the resized image
                x1 = int(x1/resize_img)
                y1 = int(y1/resize_img)
                x2 = int(x2/resize_img)
                y2 = int(y2/resize_img)    
                                 
                # here the the data to reproject (the data we need) are prepared as output of this function
                detect_object = [x1,y1,x2-x1,y2-y1]
                person.append(detect_object)  

                x1_ = str(x1)
                y1_ = str(y1)
                x2_ = str(x2)
                y2_ = str(y2)                

                print('box at index: ' + str(box_index) + ' : ClassID: ' + LABELS[int(output[base_index + 1])] + '  '
                'Confidence: ' + str(output[base_index + 2]*100) + '%  ' +
                'Top Left: (' + x1_ + ', ' + y1_ + ')  Bottom Right: (' + x2_ + ', ' + y2_ + ')')

                # overlay boxes and labels on the original image to classify
                overlay_on_image(image_to_classify, output[base_index:base_index + 7])
    return person


# overlays the boxes onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info):

    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
#    class_id = object_info[base_index + 1]
#    percentage = int(object_info[base_index + 2] * 100)
#    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    
    box_left = int((object_info[base_index + 3] * source_image_width)/1)
    box_top = int((object_info[base_index + 4] * source_image_height)/1)
    box_right = int((object_info[base_index + 5] * source_image_width)/1)
    box_bottom = int((object_info[base_index + 6] * source_image_height)/1)

    box_color = (255, 255, 0)  # box color
    box_thickness = 6
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)



# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):

    # scale the image
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img



def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    
def increment_heatmap_value(img, rects, matrix_h, resize_val):
    for x, y, w, h in rects:               
        point_source = np.array([[(x + w/2), (y+h)]], dtype='float32')
        point_source = np.array([point_source])
        point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
        image_max_y_dimension,image_max_x_dimension,_ = img.shape
        point_dest[0][0][0] = point_dest[0][0][0]/resize_val
        point_dest[0][0][1] = point_dest[0][0][1]/resize_val
        new_x = max(0,point_dest[0][0][0])
        new_x = min(image_max_x_dimension,point_dest[0][0][0])
        new_x = int(new_x)
        new_y = max(0,point_dest[0][0][1])
        new_y = min(image_max_y_dimension,point_dest[0][0][1])    
        new_y =  int(new_y)                           
        img[new_y-1, new_x-1] = img[new_y-1, new_x-1]+1
        
        
def draw_homography_point(img, rects, matrix_h, thickness = 3):
    for x, y, w, h in rects:      
        point_source = np.array([[(x+ w/2), (y+h)]], dtype='float32')
        point_source = np.array([point_source])
        point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
        image_max_y_dimension,image_max_x_dimension,_ = img.shape
        new_x = max(0,point_dest[0][0][0])
        new_x = min(image_max_x_dimension,point_dest[0][0][0])
        new_y = max(0,point_dest[0][0][1])
        new_y = min(image_max_y_dimension,point_dest[0][0][1]) 
        cv2.circle(img, (new_x, new_y), 3, (0,0,255), -1 )  

        
        
def convert_homography_point(x, y, w, h, matrix_h):
    point_source = np.array([[(x + w/2), (y+h)]], dtype='float32')
    point_source = np.array([point_source])
    point_dest = cv2.perspectiveTransform(point_source, matrix_h) 
    return point_dest[0][0][0], point_dest[0][0][1]         


def rescale_heatmap_image_value(img):
    # redefine the value for the heatmat when the max value tend to exced the 8bit
    if(img.max()==255):
        img = (img/(img.max()*1.0))*255.0
        img = img.astype(np.uint8)
    return img        
    


if __name__ == '__main__':
    
    #################################################### 
    # parameters used for the imahe resize
    #################################################### 
    resize_img = 3 # the resize to the acquired image. The HOG will be evaluated on the resized image
    additional_resize_point = 1.5 # the point in the image (source point for the homography) have been taken with resize_img=2. In case of
                                # other resize change this vale (example 2,1 4,2 .....)
    cell_heatmap_step = 20
    zoom_heatmap = 4.0
    
    #################################################### 
    # creation tracker, transformer. reproject objects
    #################################################### 
    ct = CentroidTracker()
    transformer = CoordinateTransform()
    reprojecter = ReprojectTo4326()
    
    #################################################### 
    # open and read the parameters available into the configuration json file
    #################################################### 
    with open('configReal.json') as json_data_file:
        data = json.load(json_data_file)  
    #points in source and destination images to create the Homography transformation
    pt_0_pix_src_image_X = data['config']['pt_0_pix_src_image_X']
    pt_0_pix_src_image_Y = data['config']['pt_0_pix_src_image_Y']
    pt_0_pix_dst_image_X = data['config']['pt_0_pix_dst_image_X']
    pt_0_pix_dst_image_Y = data['config']['pt_0_pix_dst_image_Y']	
    pt_1_pix_src_image_X = data['config']['pt_1_pix_src_image_X']
    pt_1_pix_src_image_Y = data['config']['pt_1_pix_src_image_Y']
    pt_1_pix_dst_image_X = data['config']['pt_1_pix_dst_image_X']
    pt_1_pix_dst_image_Y = data['config']['pt_1_pix_dst_image_Y']	
    pt_2_pix_src_image_X = data['config']['pt_2_pix_src_image_X']
    pt_2_pix_src_image_Y = data['config']['pt_2_pix_src_image_Y']
    pt_2_pix_dst_image_X = data['config']['pt_2_pix_dst_image_X']
    pt_2_pix_dst_image_Y = data['config']['pt_2_pix_dst_image_Y']
    pt_3_pix_src_image_X = data['config']['pt_3_pix_src_image_X']
    pt_3_pix_src_image_Y = data['config']['pt_3_pix_src_image_Y']
    pt_3_pix_dst_image_X = data['config']['pt_3_pix_dst_image_X']
    pt_3_pix_dst_image_Y = data['config']['pt_3_pix_dst_image_Y']
    pt_4_pix_src_image_X = data['config']['pt_4_pix_src_image_X']
    pt_4_pix_src_image_Y = data['config']['pt_4_pix_src_image_Y']
    pt_4_pix_dst_image_X = data['config']['pt_4_pix_dst_image_X']
    pt_4_pix_dst_image_Y = data['config']['pt_4_pix_dst_image_Y']
    #configuration of the connection to rabbitMQ    
    host_rabbitmq = data['config']['host_rabbitmq']
    host_rabbitmq_username = data['config']['host_rabbitmq_username']
    host_rabbitmq_psw = data['config']['host_rabbitmq_psw']
    #configuration key and id
    key = data['config']['tile38_key']    
    id = data['config']['tile38_id']
    #image information
    image_map_max_X = float(data['config']['backgound_image_max_X'])
    image_map_max_Y = float(data['config']['backgound_image_max_Y'])

    #################################################### 
    # print the vales read into the configuration file
    #################################################### 
    print("pt_0_pix_src_image_X: " + str(pt_0_pix_src_image_X)) 
    print("pt_0_pix_src_image_Y: " + str(pt_0_pix_src_image_Y))    
    print("pt_0_pix_dst_image_X: " + str(pt_0_pix_dst_image_X))    
    print("pt_0_pix_dst_image_Y: " + str(pt_0_pix_dst_image_Y))	    
    print("pt_1_pix_src_image_X: " + str(pt_1_pix_src_image_X))    
    print("pt_1_pix_src_image_Y: " + str(pt_1_pix_src_image_Y))    
    print("pt_1_pix_dst_image_X: " + str(pt_1_pix_src_image_X))    	
    print("pt_1_pix_dst_image_Y: " + str(pt_1_pix_dst_image_Y))   
    print("pt_2_pix_src_image_X: " + str(pt_2_pix_src_image_X))   
    print("pt_2_pix_src_image_Y: " + str(pt_2_pix_src_image_Y))   
    print("pt_2_pix_dst_image_X: " + str(pt_2_pix_dst_image_X))   
    print("pt_2_pix_dst_image_Y: " + str(pt_2_pix_dst_image_Y))   
    print("pt_3_pix_src_image_X: " + str(pt_3_pix_src_image_X))   
    print("pt_3_pix_src_image_Y: " + str(pt_3_pix_src_image_Y))  
    print("pt_3_pix_dst_image_X: " + str(pt_3_pix_dst_image_X))   
    print("pt_3_pix_dst_image_Y: " + str(pt_3_pix_dst_image_Y))    
    print("pt_4_pix_src_image_X: " + str(pt_4_pix_src_image_X))    
    print("pt_4_pix_src_image_Y: " + str(pt_4_pix_src_image_Y))    
    print("pt_4_pix_dst_image_X: " + str(pt_4_pix_dst_image_X))    
    print("pt_4_pix_dst_image_Y: " + str(pt_4_pix_dst_image_Y))  
    print("host_rabbitmq: " + host_rabbitmq)
    print("host_rabbitmq_username: " + host_rabbitmq_username)    
    print("host_rabbitmq_psw: " + host_rabbitmq_psw)
    print("key: " + key)
    print("id: " + id)
    print("image_map_max_X: " + str(image_map_max_X))
    print("image_map_max_Y: " + str(image_map_max_Y))
    
    #################################################### 
    # Create a sequence of points to make a contour of a valid detection region. This for the problem of point detected in region
    # where is not possible that peaple ore detected (outside a valid region)
    #################################################### 
#    valid_polygon = [None]*4
#    valid_polygon[0] = (1007, 1388)
#    valid_polygon[1] = (1007, 720)
#    valid_polygon[2] = (1487, 720)
#    valid_polygon[3] = (1487, 1390)
#    polygon = Polygon([valid_polygon[0], valid_polygon[1], valid_polygon[2], valid_polygon[3]])

    ####################################################  
    # connect to rabbitmq
    ####################################################
    credentials = pika.PlainCredentials(host_rabbitmq_username, host_rabbitmq_psw)
    #parameters = pika.ConnectionParameters(host_rabbitmq,5672,'/',credentials, socket_timeout=10000000, heartbeat_interval=0,blocked_connection_timeout=300)
    parameters = pika.ConnectionParameters(host_rabbitmq,5672,'/',credentials)
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()    


    ####################################################  
    # load the background image and prepare frames
    ####################################################
    img_map = cv2.imread('mapReal.png')    
    rows_map_frame,cols_map_frame, _ = img_map.shape 
    img_map_view = np.zeros((int(cols_map_frame/4), int(rows_map_frame/4), 3), np.uint8)
    heatmap_gray = np.zeros((int(rows_map_frame/cell_heatmap_step), int(cols_map_frame/cell_heatmap_step), 1), dtype = np.uint8)
 
    ####################################################  
    # Find the homography matrix 'h'
    # https://zbigatron.com/mapping-camera-coordinates-to-a-2d-floor-plan/
    ####################################################    
    pts_src = np.array([[pt_0_pix_src_image_X/additional_resize_point, pt_0_pix_src_image_Y/additional_resize_point], [pt_1_pix_src_image_X/additional_resize_point, pt_1_pix_src_image_Y/additional_resize_point], [pt_2_pix_src_image_X/additional_resize_point, pt_2_pix_src_image_Y/additional_resize_point],[pt_3_pix_src_image_X/additional_resize_point, pt_3_pix_src_image_Y/additional_resize_point], [pt_4_pix_src_image_X/additional_resize_point, pt_4_pix_src_image_Y/additional_resize_point]])
    pts_dst = np.array([[pt_0_pix_dst_image_X, pt_0_pix_dst_image_Y], [pt_1_pix_dst_image_X, pt_1_pix_dst_image_Y], [pt_2_pix_dst_image_X, pt_2_pix_dst_image_Y],[pt_3_pix_dst_image_X, pt_3_pix_dst_image_Y], [pt_4_pix_dst_image_X, pt_4_pix_dst_image_Y]])
    h, status = cv2.findHomography(pts_src, pts_dst) # # calculate matrix H
    cv2.namedWindow('homography')

    ####################################################  
    # get the dinonsion of the camera frame
    #################################################### 
    cap=cv2.VideoCapture("http://root:progtrl01@192.168.208.200/mjpg/1/video.mjpg")
    #cap=cv2.VideoCapture("http://admin:admin@192.168.208.200/jpg/image.jpg?size=3")
    _,input_image=cap.read() # acquire a new image
    height, width, _ = input_image.shape 
    width = np.array(width, dtype=float)
    height = np.array(height, dtype=float)
    #Assign the shape of the input image to image_shape variable
    image_shape = (height,width)    
    
    ####################################################
    # Get a list of ALL the sticks that are plugged in we need at least one
    ####################################################
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
        
    ####################################################
    # Pick the first stick to run the network
    ####################################################
    device = mvnc.Device(devices[0])
    
    ####################################################
    # Open the NCS
    ####################################################
    device.OpenDevice()
    
    ####################################################
    # The graph file that was created with the ncsdk compiler
    ####################################################
    graph_file_name = 'graph'

    ####################################################
    # read in the graph file to memory buffer
    ####################################################
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    ####################################################
    # create the NCAPI graph instance from the memory buffer containing the graph file.
    ####################################################
    graph = device.AllocateGraph(graph_in_memory)    

    
    while True:
        ####################################################
        # read a frame from the camera
        ####################################################
        print("")
        print("******************** READ NEW FRAME ********************")
        _,frame_camera = cap.read()
        
        a = datetime.datetime.now()
        
        ## resize the original image
        #frame = cv2.resize(frame_camera, (0,0), fx=1.0/resize_img, fy=1.0/resize_img) x

        # run a single inference on the image
        detected_box = run_inference(frame_camera, graph) 
        print("detected_box ", detected_box)
        
        # resize the original image
        frame = cv2.resize(frame_camera, (0,0), fx=1.0/resize_img, fy=1.0/resize_img) 
        
        b = datetime.datetime.now()
        print("Elaboration time: ",b-a)                            
    
        draw_homography_point(img_map, detected_box, h)
        
        increment_heatmap_value(heatmap_gray, detected_box, h, cell_heatmap_step) # increment vale for the heatmap in the gray image       
        heatmap_gray = rescale_heatmap_image_value(heatmap_gray)        
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_color_resize_big = cv2.resize(heatmap_color, (0,0), fx=zoom_heatmap, fy=zoom_heatmap)

        rects_homography = []
        rects_lat_lon = []
        rects_homography_and_lat_lon = []
        
        #convert data in geographic position and provide the data to rabbitmq
        for x, y, dim_w, dim_h  in detected_box:             
            homographyX_in_pixel, homographyY_in_pixel = convert_homography_point(x, y, dim_w, dim_h, h)          
            box_h = [int(homographyX_in_pixel), int(homographyY_in_pixel), int(dim_w), int(dim_h)]
            homographyY_in_pixel = image_map_max_Y - homographyY_in_pixel
            rects_homography.append(box_h)
            
            #point_inside = polygon.contains(Point(homographyX_in_pixel,image_map_max_Y - homographyY_in_pixel))
            #if(point_inside==True):
            
            metersX,metersY = transformer.pixelToMeter(homographyX_in_pixel,homographyY_in_pixel,image_map_max_X,image_map_max_Y)  
            newx,newy = transformer.transform(metersX,metersY)       
            newy,newx = reprojecter.MetersToLatLon(newx,newy)  
            box_lat_lon = [newy, newx]           
            rects_lat_lon.append(box_lat_lon)

                              
        objects = ct.update2(rects_lat_lon)             
               
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {} - {} {}".format(objectID, centroid[1], centroid[0])
            print(text)
            # insert data in queue in rabbitmq 
            
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            timezone =time.strftime("%z")
            timestamp = st+timezone            
            
            #body = '{"name":"' + str(objectID) + '","timestamp":"2018-10-19T12:46:50.985+0200","geometry":{"type":"Point","coordinates":[' + str(centroid[1]) + ',' + str(centroid[0]) + ']},"accuracy":0.8, "source":{"type":"Manual","name":"PythonClientCameraRD"},"extra":{"Tile38Key":"' + key + '","SoftwareVersion":"1.0-SNAPSHOT"}}'            
            body = '{"name":"' + str(objectID) + '","timestamp":"' + timestamp + '","geometry":{"type":"Point","coordinates":[' + str(centroid[1]) + ',' + str(centroid[0]) + ']},"accuracy":0.8, "source":{"type":"Manual","name":"PythonClientCameraRD"},"extra":{"Tile38Key":"' + key + '","SoftwareVersion":"cameraYOLO"}}'            
            #print(body)
            channel.basic_publish(exchange='trilogis_exchange_pos',routing_key='trilogis_position',body=body, properties=pika.BasicProperties(delivery_mode = 2)) # make message persistent
    
            
        cv2.imshow('feed',frame)  
        cv2.imshow('heatmap',heatmap_color_resize_big)          
        img_map_view = cv2.resize(img_map, (int(cols_map_frame/4), int(rows_map_frame/4)))
        cv2.imshow('homography',img_map_view)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    res = connection.close()
    print ('rabbitmq connection close - ' + str(res))  
    
    ##########################################
    # Clean up the graph and the device
    ##########################################
    graph.DeallocateGraph()
    device.CloseDevice()
    cv2.destroyAllWindows()
    
