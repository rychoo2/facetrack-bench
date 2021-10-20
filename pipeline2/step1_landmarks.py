import os
from libs.utils import get_timestamp, get_datasets
import subprocess
import cv2
import shutil
import pandas as pd

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data2"
output_path = "{}/landmarks/{}".format(train_data_dir,  get_timestamp())
openface_bin_path = os.environ.get('OPENFACE_BIN_PATH')


def run_live_openface_feature_extraction(output_path):
    openface_output_path = output_path + "/openface"

    ## 1. Execute the process in the background, below is the foreground version

    # subprocess.check_call([
    #     openface_bin_path + "/FeatureExtraction",
    #     "-verbose",
    #     "-device", "0",
    #     "-oc", "H264",
    #     "-out_dir", openface_output_path
    # ])

    ## 2. read header from the output file
    header = "frame,face_id,timestamp,confidence,success,gaze_0_x,gaze_0_y,gaze_0_z,gaze_1_x,gaze_1_y,gaze_1_z,gaze_angle_x,gaze_angle_y,eye_lmk_x_0,eye_lmk_x_1,eye_lmk_x_2,eye_lmk_x_3,eye_lmk_x_4,eye_lmk_x_5,eye_lmk_x_6,eye_lmk_x_7,eye_lmk_x_8,eye_lmk_x_9,eye_lmk_x_10,eye_lmk_x_11,eye_lmk_x_12,eye_lmk_x_13,eye_lmk_x_14,eye_lmk_x_15,eye_lmk_x_16,eye_lmk_x_17,eye_lmk_x_18,eye_lmk_x_19,eye_lmk_x_20,eye_lmk_x_21,eye_lmk_x_22,eye_lmk_x_23,eye_lmk_x_24,eye_lmk_x_25,eye_lmk_x_26,eye_lmk_x_27,eye_lmk_x_28,eye_lmk_x_29,eye_lmk_x_30,eye_lmk_x_31,eye_lmk_x_32,eye_lmk_x_33,eye_lmk_x_34,eye_lmk_x_35,eye_lmk_x_36,eye_lmk_x_37,eye_lmk_x_38,eye_lmk_x_39,eye_lmk_x_40,eye_lmk_x_41,eye_lmk_x_42,eye_lmk_x_43,eye_lmk_x_44,eye_lmk_x_45,eye_lmk_x_46,eye_lmk_x_47,eye_lmk_x_48,eye_lmk_x_49,eye_lmk_x_50,eye_lmk_x_51,eye_lmk_x_52,eye_lmk_x_53,eye_lmk_x_54,eye_lmk_x_55,eye_lmk_y_0,eye_lmk_y_1,eye_lmk_y_2,eye_lmk_y_3,eye_lmk_y_4,eye_lmk_y_5,eye_lmk_y_6,eye_lmk_y_7,eye_lmk_y_8,eye_lmk_y_9,eye_lmk_y_10,eye_lmk_y_11,eye_lmk_y_12,eye_lmk_y_13,eye_lmk_y_14,eye_lmk_y_15,eye_lmk_y_16,eye_lmk_y_17,eye_lmk_y_18,eye_lmk_y_19,eye_lmk_y_20,eye_lmk_y_21,eye_lmk_y_22,eye_lmk_y_23,eye_lmk_y_24,eye_lmk_y_25,eye_lmk_y_26,eye_lmk_y_27,eye_lmk_y_28,eye_lmk_y_29,eye_lmk_y_30,eye_lmk_y_31,eye_lmk_y_32,eye_lmk_y_33,eye_lmk_y_34,eye_lmk_y_35,eye_lmk_y_36,eye_lmk_y_37,eye_lmk_y_38,eye_lmk_y_39,eye_lmk_y_40,eye_lmk_y_41,eye_lmk_y_42,eye_lmk_y_43,eye_lmk_y_44,eye_lmk_y_45,eye_lmk_y_46,eye_lmk_y_47,eye_lmk_y_48,eye_lmk_y_49,eye_lmk_y_50,eye_lmk_y_51,eye_lmk_y_52,eye_lmk_y_53,eye_lmk_y_54,eye_lmk_y_55,eye_lmk_X_0,eye_lmk_X_1,eye_lmk_X_2,eye_lmk_X_3,eye_lmk_X_4,eye_lmk_X_5,eye_lmk_X_6,eye_lmk_X_7,eye_lmk_X_8,eye_lmk_X_9,eye_lmk_X_10,eye_lmk_X_11,eye_lmk_X_12,eye_lmk_X_13,eye_lmk_X_14,eye_lmk_X_15,eye_lmk_X_16,eye_lmk_X_17,eye_lmk_X_18,eye_lmk_X_19,eye_lmk_X_20,eye_lmk_X_21,eye_lmk_X_22,eye_lmk_X_23,eye_lmk_X_24,eye_lmk_X_25,eye_lmk_X_26,eye_lmk_X_27,eye_lmk_X_28,eye_lmk_X_29,eye_lmk_X_30,eye_lmk_X_31,eye_lmk_X_32,eye_lmk_X_33,eye_lmk_X_34,eye_lmk_X_35,eye_lmk_X_36,eye_lmk_X_37,eye_lmk_X_38,eye_lmk_X_39,eye_lmk_X_40,eye_lmk_X_41,eye_lmk_X_42,eye_lmk_X_43,eye_lmk_X_44,eye_lmk_X_45,eye_lmk_X_46,eye_lmk_X_47,eye_lmk_X_48,eye_lmk_X_49,eye_lmk_X_50,eye_lmk_X_51,eye_lmk_X_52,eye_lmk_X_53,eye_lmk_X_54,eye_lmk_X_55,eye_lmk_Y_0,eye_lmk_Y_1,eye_lmk_Y_2,eye_lmk_Y_3,eye_lmk_Y_4,eye_lmk_Y_5,eye_lmk_Y_6,eye_lmk_Y_7,eye_lmk_Y_8,eye_lmk_Y_9,eye_lmk_Y_10,eye_lmk_Y_11,eye_lmk_Y_12,eye_lmk_Y_13,eye_lmk_Y_14,eye_lmk_Y_15,eye_lmk_Y_16,eye_lmk_Y_17,eye_lmk_Y_18,eye_lmk_Y_19,eye_lmk_Y_20,eye_lmk_Y_21,eye_lmk_Y_22,eye_lmk_Y_23,eye_lmk_Y_24,eye_lmk_Y_25,eye_lmk_Y_26,eye_lmk_Y_27,eye_lmk_Y_28,eye_lmk_Y_29,eye_lmk_Y_30,eye_lmk_Y_31,eye_lmk_Y_32,eye_lmk_Y_33,eye_lmk_Y_34,eye_lmk_Y_35,eye_lmk_Y_36,eye_lmk_Y_37,eye_lmk_Y_38,eye_lmk_Y_39,eye_lmk_Y_40,eye_lmk_Y_41,eye_lmk_Y_42,eye_lmk_Y_43,eye_lmk_Y_44,eye_lmk_Y_45,eye_lmk_Y_46,eye_lmk_Y_47,eye_lmk_Y_48,eye_lmk_Y_49,eye_lmk_Y_50,eye_lmk_Y_51,eye_lmk_Y_52,eye_lmk_Y_53,eye_lmk_Y_54,eye_lmk_Y_55,eye_lmk_Z_0,eye_lmk_Z_1,eye_lmk_Z_2,eye_lmk_Z_3,eye_lmk_Z_4,eye_lmk_Z_5,eye_lmk_Z_6,eye_lmk_Z_7,eye_lmk_Z_8,eye_lmk_Z_9,eye_lmk_Z_10,eye_lmk_Z_11,eye_lmk_Z_12,eye_lmk_Z_13,eye_lmk_Z_14,eye_lmk_Z_15,eye_lmk_Z_16,eye_lmk_Z_17,eye_lmk_Z_18,eye_lmk_Z_19,eye_lmk_Z_20,eye_lmk_Z_21,eye_lmk_Z_22,eye_lmk_Z_23,eye_lmk_Z_24,eye_lmk_Z_25,eye_lmk_Z_26,eye_lmk_Z_27,eye_lmk_Z_28,eye_lmk_Z_29,eye_lmk_Z_30,eye_lmk_Z_31,eye_lmk_Z_32,eye_lmk_Z_33,eye_lmk_Z_34,eye_lmk_Z_35,eye_lmk_Z_36,eye_lmk_Z_37,eye_lmk_Z_38,eye_lmk_Z_39,eye_lmk_Z_40,eye_lmk_Z_41,eye_lmk_Z_42,eye_lmk_Z_43,eye_lmk_Z_44,eye_lmk_Z_45,eye_lmk_Z_46,eye_lmk_Z_47,eye_lmk_Z_48,eye_lmk_Z_49,eye_lmk_Z_50,eye_lmk_Z_51,eye_lmk_Z_52,eye_lmk_Z_53,eye_lmk_Z_54,eye_lmk_Z_55,pose_Tx,pose_Ty,pose_Tz,pose_Rx,pose_Ry,pose_Rz,x_0,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9,x_10,x_11,x_12,x_13,x_14,x_15,x_16,x_17,x_18,x_19,x_20,x_21,x_22,x_23,x_24,x_25,x_26,x_27,x_28,x_29,x_30,x_31,x_32,x_33,x_34,x_35,x_36,x_37,x_38,x_39,x_40,x_41,x_42,x_43,x_44,x_45,x_46,x_47,x_48,x_49,x_50,x_51,x_52,x_53,x_54,x_55,x_56,x_57,x_58,x_59,x_60,x_61,x_62,x_63,x_64,x_65,x_66,x_67,y_0,y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8,y_9,y_10,y_11,y_12,y_13,y_14,y_15,y_16,y_17,y_18,y_19,y_20,y_21,y_22,y_23,y_24,y_25,y_26,y_27,y_28,y_29,y_30,y_31,y_32,y_33,y_34,y_35,y_36,y_37,y_38,y_39,y_40,y_41,y_42,y_43,y_44,y_45,y_46,y_47,y_48,y_49,y_50,y_51,y_52,y_53,y_54,y_55,y_56,y_57,y_58,y_59,y_60,y_61,y_62,y_63,y_64,y_65,y_66,y_67,X_0,X_1,X_2,X_3,X_4,X_5,X_6,X_7,X_8,X_9,X_10,X_11,X_12,X_13,X_14,X_15,X_16,X_17,X_18,X_19,X_20,X_21,X_22,X_23,X_24,X_25,X_26,X_27,X_28,X_29,X_30,X_31,X_32,X_33,X_34,X_35,X_36,X_37,X_38,X_39,X_40,X_41,X_42,X_43,X_44,X_45,X_46,X_47,X_48,X_49,X_50,X_51,X_52,X_53,X_54,X_55,X_56,X_57,X_58,X_59,X_60,X_61,X_62,X_63,X_64,X_65,X_66,X_67,Y_0,Y_1,Y_2,Y_3,Y_4,Y_5,Y_6,Y_7,Y_8,Y_9,Y_10,Y_11,Y_12,Y_13,Y_14,Y_15,Y_16,Y_17,Y_18,Y_19,Y_20,Y_21,Y_22,Y_23,Y_24,Y_25,Y_26,Y_27,Y_28,Y_29,Y_30,Y_31,Y_32,Y_33,Y_34,Y_35,Y_36,Y_37,Y_38,Y_39,Y_40,Y_41,Y_42,Y_43,Y_44,Y_45,Y_46,Y_47,Y_48,Y_49,Y_50,Y_51,Y_52,Y_53,Y_54,Y_55,Y_56,Y_57,Y_58,Y_59,Y_60,Y_61,Y_62,Y_63,Y_64,Y_65,Y_66,Y_67,Z_0,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6,Z_7,Z_8,Z_9,Z_10,Z_11,Z_12,Z_13,Z_14,Z_15,Z_16,Z_17,Z_18,Z_19,Z_20,Z_21,Z_22,Z_23,Z_24,Z_25,Z_26,Z_27,Z_28,Z_29,Z_30,Z_31,Z_32,Z_33,Z_34,Z_35,Z_36,Z_37,Z_38,Z_39,Z_40,Z_41,Z_42,Z_43,Z_44,Z_45,Z_46,Z_47,Z_48,Z_49,Z_50,Z_51,Z_52,Z_53,Z_54,Z_55,Z_56,Z_57,Z_58,Z_59,Z_60,Z_61,Z_62,Z_63,Z_64,Z_65,Z_66,Z_67,p_scale,p_rx,p_ry,p_rz,p_tx,p_ty,p_0,p_1,p_2,p_3,p_4,p_5,p_6,p_7,p_8,p_9,p_10,p_11,p_12,p_13,p_14,p_15,p_16,p_17,p_18,p_19,p_20,p_21,p_22,p_23,p_24,p_25,p_26,p_27,p_28,p_29,p_30,p_31,p_32,p_33,AU01_r,AU02_r,AU04_r,AU05_r,AU06_r,AU07_r,AU09_r,AU10_r,AU12_r,AU14_r,AU15_r,AU17_r,AU20_r,AU23_r,AU25_r,AU26_r,AU45_r,AU01_c,AU02_c,AU04_c,AU05_c,AU06_c,AU07_c,AU09_c,AU10_c,AU12_c,AU14_c,AU15_c,AU17_c,AU20_c,AU23_c,AU25_c,AU26_c,AU28_c,AU45_c"
    header_names = header.split(',')

    try:
        while True:
            ## 3. yield actual landmarks generated by openface,
            ## read last full line from the output file, do not return partial line
            ## do not return same line twice
            landmarks = '1,0,0.000,0.98,1,0.315830,0.401336,-0.859756,0.162663,0.383982,-0.908900,0.264,0.418,655.7,658.1,665.1,672.7,676.3,674.4,666.9,659.3,642.0,646.7,654.1,663.2,671.8,677.8,681.8,677.3,670.6,662.8,654.9,647.6,662.9,666.1,669.0,670.0,668.5,665.4,662.4,661.4,762.5,764.2,770.6,778.1,782.2,780.6,774.1,766.0,753.3,757.6,763.7,771.2,778.2,783.7,787.8,784.5,779.5,773.2,766.0,758.8,769.2,772.7,775.7,776.5,774.6,771.1,768.1,767.3,152.9,145.0,141.0,143.2,150.4,158.6,162.2,160.0,155.9,151.9,149.2,147.4,148.1,150.8,155.1,158.0,159.8,160.9,160.8,159.2,154.9,155.8,154.1,150.9,147.9,147.0,148.6,151.9,140.3,132.4,128.3,130.4,137.4,145.3,149.4,148.0,144.6,139.7,136.0,134.1,134.5,136.0,139.2,143.3,146.0,147.6,147.8,147.2,143.0,144.0,142.1,138.4,135.1,134.1,136.0,139.7,8.4,9.7,13.5,17.6,19.6,18.6,14.5,10.4,1.1,3.6,7.6,12.4,17.1,20.5,22.7,20.2,16.5,12.3,8.0,4.1,12.4,14.1,15.7,16.2,15.4,13.7,12.1,11.6,71.4,72.5,76.5,81.1,83.6,82.4,78.4,73.5,66.2,68.6,72.0,76.5,80.8,84.5,87.4,85.0,81.7,77.7,73.4,69.3,75.6,77.7,79.6,80.1,79.0,76.9,75.0,74.5,-111.5,-115.6,-117.8,-116.9,-113.3,-109.1,-107.0,-107.9,-111.1,-112.5,-113.3,-114.0,-114.0,-113.1,-111.4,-109.5,-108.1,-107.4,-107.6,-108.8,-110.8,-110.4,-111.4,-113.1,-114.6,-115.0,-114.1,-112.3,-128.1,-132.9,-135.7,-134.8,-130.8,-125.9,-123.1,-123.7,-125.8,-128.4,-130.4,-131.6,-131.9,-131.7,-130.7,-127.6,-125.2,-123.8,-123.6,-124.1,-127.0,-126.6,-127.9,-130.1,-132.0,-132.4,-131.1,-128.9,471.1,470.5,470.9,471.9,473.0,473.9,473.2,472.2,476.1,472.9,470.4,469.3,470.7,473.1,475.8,474.3,472.5,471.9,472.5,474.1,472.9,473.3,473.5,473.2,472.8,472.3,472.2,472.4,509.9,510.8,512.4,513.8,514.1,513.2,511.6,510.4,511.1,510.0,509.5,509.9,511.8,514.8,517.7,514.9,512.1,510.1,509.5,510.2,512.0,512.6,513.3,513.8,513.6,513.0,512.2,511.8,46.6,-82.8,507.1,0.335,-0.216,-0.047,577.7,582.4,589.5,598.2,612.5,633.1,656.4,683.4,715.1,743.5,763.1,779.5,793.4,803.1,808.0,810.5,809.8,608.5,623.4,647.7,674.3,698.4,744.6,761.6,778.6,795.2,806.1,724.2,729.1,734.1,739.2,707.3,721.2,735.0,747.0,756.6,639.6,654.6,670.8,684.3,670.9,655.0,752.2,763.2,778.0,788.1,779.7,765.4,691.7,710.5,725.2,734.7,744.3,754.3,763.7,756.8,746.9,736.7,726.7,711.7,700.3,725.1,734.7,744.4,759.0,745.4,735.5,725.7,151.1,185.4,218.3,249.0,278.1,301.6,319.3,335.0,340.6,337.7,321.8,299.8,272.6,242.7,210.7,177.2,143.2,131.1,113.7,106.0,107.5,114.0,108.1,97.9,92.6,93.8,105.0,144.2,168.4,191.8,215.6,230.4,234.8,237.9,233.0,226.7,156.6,148.1,146.8,153.6,159.4,161.3,146.0,135.6,134.3,140.0,146.9,148.6,280.1,272.3,267.6,269.6,265.6,267.6,270.6,281.0,286.7,289.0,289.0,286.6,279.8,276.3,276.2,273.9,272.4,274.8,277.3,277.1,-36.8,-34.2,-30.3,-25.3,-16.6,-4.1,9.7,25.4,44.0,61.8,76.0,89.0,99.2,105.6,108.6,110.0,109.8,-17.6,-9.1,4.2,18.6,31.6,58.1,68.5,79.3,90.2,98.2,46.6,48.8,51.0,53.1,37.1,44.7,52.3,59.2,65.1,-0.2,8.1,17.1,24.7,17.2,8.3,64.3,70.9,80.0,87.0,81.1,72.3,29.3,39.5,47.6,53.0,58.7,65.4,72.5,66.8,60.2,54.1,48.4,40.2,34.0,47.7,53.2,59.0,69.2,59.4,53.5,47.9,-123.3,-103.7,-84.9,-67.1,-49.5,-35.1,-24.2,-14.7,-11.3,-13.3,-23.6,-38.4,-56.5,-76.0,-96.5,-117.9,-140.1,-127.7,-135.7,-138.7,-137.1,-133.2,-139.9,-147.6,-153.1,-154.8,-150.7,-119.3,-104.9,-91.1,-77.3,-71.5,-68.9,-67.2,-70.3,-74.4,-113.3,-117.2,-118.2,-115.0,-111.6,-110.1,-122.6,-129.1,-130.9,-129.3,-123.8,-121.8,-45.3,-49.1,-51.6,-50.6,-53.1,-52.9,-52.4,-45.2,-41.3,-39.7,-39.7,-41.1,-45.3,-46.9,-47.0,-48.6,-50.9,-48.0,-46.3,-46.4,516.4,519.5,524.4,528.9,529.1,525.3,519.2,512.2,512.2,522.2,540.5,557.9,566.0,566.6,565.6,564.2,565.6,488.3,482.0,477.9,475.2,473.7,485.8,492.8,500.9,508.7,517.0,484.0,479.0,473.7,468.5,482.9,481.6,481.4,484.4,488.3,487.5,484.2,485.1,487.8,486.5,485.0,501.2,503.6,507.3,514.3,508.3,504.4,495.9,490.2,488.8,489.9,492.3,501.2,513.0,500.6,492.8,489.9,488.8,490.3,493.8,490.2,491.1,494.1,508.7,492.9,490.2,489.6,1.740,0.194,-0.328,-0.017,720.093,215.410,-7.029,-9.064,-28.626,16.371,8.706,8.382,2.475,12.243,5.777,8.384,-12.453,8.933,4.132,-1.507,-1.041,-4.893,-0.667,-7.031,-8.131,-0.318,5.958,3.802,4.856,-1.781,4.410,1.884,1.055,-1.575,0.157,0.167,0.647,0.083,0.223,-0.028,0.53,0.00,0.00,0.00,0.38,0.00,0.00,0.31,0.24,0.00,0.06,0.81,0.00,0.00,0.69,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00'
            landmarks_dict = dict(zip(header_names, landmarks.split(',')))
            yield landmarks_dict

    except GeneratorExit:
        ## 4. kill the background process here
        print("finished")
        raise


def run_openface_feature_extraction(input_path, output_path):
    openface_output_path = output_path + "/openface"


    subprocess.check_call([
        openface_bin_path +"/FeatureExtraction",
        "-verbose",
        "-fdir", input_path + "/images",
        "-oc H264",
        "-out_dir", openface_output_path
    ])
    extract_images_from_video(openface_output_path + "/images.avi")
    shutil.move(openface_output_path+"/images", output_path)
    landmarks_csv = openface_output_path+"/images.csv"

    df = pd.read_csv(landmarks_csv)

    df.insert(1, 'landmark_image', df.apply(
        lambda row: image_filename(row['frame']),
        axis=1)
    )

    df.to_csv(output_path+"/landmarks.csv", index=False)


def extract_images_from_video(videofile):
    output_path = os.path.dirname(videofile)
    os.makedirs(output_path + "/images")
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(output_path, image_filename(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count+=1

def image_filename(frame):
    return f"images/frame_{int(frame):d}.jpg"

def generate_landmarks_for_datasets(input_root, output_root):
    path, datasets = get_datasets(input_root)
    for dataset in datasets:
        run_openface_feature_extraction("{}/{}".format(path, dataset), "{}/{}".format(output_root, dataset))


if __name__ == '__main__':

    generate_landmarks_for_datasets(train_data_dir, output_path)

