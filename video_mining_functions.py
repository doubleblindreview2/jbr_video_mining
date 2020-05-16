#loop through video and get brightness and colors
def color_loop(input_path, output_path,name, skip_frame = 5):
    cap = cv2.VideoCapture(input_path)
    col_stats = pd.DataFrame(columns = ['frame_num','colorfulness','saturation','value','black','blue','brown','grey','green','orange','pink','purple','red','white','yellow'])
    
    try:
        while(True):
            ret, frame = cap.read()
            if not(ret): break

            #get time and frame count and all color frames
            time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if frame_count %skip_frame == 0:
                
                rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

                #get resolution, aspect ratio
                resolution = [frame.shape[0],frame.shape[1]]

                # get values for saturation, value, colorfulness
                saturation,value = basic_cols(hsv_frame)
                colorfulness = image_colorfulness(rgb_frame)
                
                # get exact colors
                img_vec=np.reshape(rgb_frame, (-1, 3))
                black,blue,brown,grey,green,orange,pink,purple,red,white,yellow = get_color_share(w2c,img_vec)
                col_stats = col_stats.append(pd.DataFrame([[frame_count,saturation,value,colorfulness,black,blue,brown,grey,green,orange,pink,purple,red,white,yellow]], columns = ['frame_num','colorfulness','saturation','value','black','blue','brown','grey','green','orange','pink','purple','red','white','yellow']), ignore_index = True)
             
            
        cap.release()
        cv2.destroyAllWindows()
        
        # here df aggregation mean
        
        write_csv(col_stats, output_path, name+'_FrameLevel_colors.csv')        
        write_csv(resolution, output_path, name+'_VideoLevel_resolution.csv')
        color_columns = ['colorfulness','saturation','value','black','blue','brown','grey','green','orange','pink','purple','red','white','yellow']

        color_columns_avg = [col+'_avg' for col in color_columns]
        avg_col_stats = pd.DataFrame([col_stats[color_columns].mean(axis=1)],columns=color_columns_avg)
        write_csv(avg_col_stats, output_path, name+'_VideoLevel_colors.csv') 
        return True
    except:
        return  sys.exc_info()[1]

#get video length
def vid_length(input_path,output_path,vid_id): 
    
    try:
        clip = VideoFileClip(input_path)
        write_csv(clip.duration,output_path,vid_id+'_VideoLevel_length.csv' )
        try: clip.reader.close()
        except: pass
        try: clip.audio.reader.close_proc()
        except: pass
        return True
    except:
        return  sys.exc_info()[1]

#get scene cuts
def get_cuts(input_path, output_path, name, save_img = False, del_row = True):
     try:
         
        #get command
        command = 'scenedetect --input ' +'"'+ input_path + '"'+ ' --output ' +'"'+output_path +'"'+' detect-content list-scenes -f ' + name + '_FrameLevel_Scenes.csv -q '
        print(command)
        if save_img: command = command + 'save-images -o ' + output_path + '/scene_imgs/'
        
        #run command
        subprocess.call(command, shell=True)
        
        #delete top row as its filled with unnecessary time code for video split
        if del_row: del_top_row_csv(str(output_path + name + '_FrameLevel_Scenes.csv'))
        
        # calculate aggregated feature avg_scene_freq 
        data = pd.read_csv(output_path + name + '_FrameLevel_Scenes.csv')
        avg_scene_freq = data['Scene Number'].max()/data['End Time (seconds)'].max()
        write_csv(avg_scene_freq,output_path,name+'_VideoLevel_avg_scene_freq.csv')
        
        
        return True
    
     except:
        return  sys.exc_info()[1]

#deletes the top rows until row index = x in file
def del_top_row_csv(file, x = 0):
    with open(file,"r") as f:
        lines = csv.reader(f,delimiter=",") # , is default
        rows = list(lines)
        del(rows[x])
    
    with open(file,'w', newline = '\n') as f2:
        cw = csv.writer(f2, delimiter = ',')
        cw.writerows(rows)

#write csv file independent of type of datainput (df, str)
def write_csv(content, folder, name, index = False, header = True, mode = 'w'):  
    try:
        content.to_csv(folder+name, index = index, header = header, mode = mode)      
    except:
        try:
            with open(str(folder)+str(name), mode = mode) as output:
                output.write(str(content))
        except:
            print('csv creation failed for: ', folder, name) 

#create a folder for each video to store extracted feature information
def create_folder(location):
    try:
        os.mkdir(location)
    except:
        pass
             
#get face information by looping through faces 
def video_loop_faces(input_path, output_path,name,skip_frame = 10):
    cap = cv2.VideoCapture(input_path)
    
    face_file = pd.DataFrame(columns = ['time','frame','num_faces','faces'])
    
    write_csv(face_file,output_path,name + '_FrameLevel_faces.csv')
    #detector = MTCNN()
    
   
    try:
        while(True):
            ret, frame = cap.read()
            if not(ret): break
                
            if cap.get(cv2.CAP_PROP_POS_FRAMES) %skip_frame == 0:
                
                # get face vector from frame
                contain  = detector.detect_faces(frame)
                count = len(contain)

                row = str(round(cap.get(cv2.CAP_PROP_POS_MSEC),0))+','+str(cap.get(cv2.CAP_PROP_POS_FRAMES))+','+str(count)+',"'+str(contain)+'"\n'
                write_csv(row,output_path,name + '_FrameLevel_faces.csv', mode = 'a')
                
#                 for i in range(0, count):  #attention for loop does run parallel to frame reading - frame changes within loop on it
#                     x_start = contain[i]['box'][0]
#                     y_start = contain[i]['box'][1]
#                     x_end = x_start + contain[i]['box'][2]
#                     y_end = y_start + contain[i]['box'][3]
#                     start_coords = (x_start, y_start)
#                     end_coords = (x_end, y_end)    
#                     frame = cv2.rectangle(frame, start_coords, end_coords, (0,255,0), 4)
#                     cv2.imwrite(frame_path+name + "_frame_%d_ms.jpg" %round(cap.get(cv2.CAP_PROP_POS_MSEC),0), frame)

        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        
        # get aggregated variables avg_face_num and face_ratio
        df = pd.read_csv(output_path+name + '_FrameLevel_faces.csv')
        face_ratio = df[df['num_faces'] != 0].shape[0] / df.shape[0]
        avg_num_faces = df[df['num_faces'] != 0]['num_faces'].mean()
        
        write_csv(face_ratio,output_path,name + '_VideoLevel_face_ratio.csv')
        write_csv(avg_num_faces,output_path,name + '_VideoLevel_avg_num_faces.csv')


        return True
    except:
        return  sys.exc_info()[1]

#format frames for input to coco
def LoadImages2(img0):
    # Padded resize
    img = letterbox(img0, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

#get 80 coco objet information
def coco_loop(input_path, output_path,name,skip_frame = 10):
    cap = cv2.VideoCapture(input_path)
    try:
        with open(output_path + name + '_FrameLevel_coco_v2.csv', 'w') as file:
            file.write('frame,y1,x1,y2,x2,object,confidence\n')    


            while(True):
                ret, frame = cap.read()
                if not(ret): break
                    
                if cap.get(cv2.CAP_PROP_POS_FRAMES) %skip_frame == 0:
                    img = LoadImages2(frame)
                    img = torch.from_numpy(img).to(device)

                    #what does this do?
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    pred = model(img)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres, iou_thres)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                            # Write results
                            

                            for *xyxy, conf, cls in det:
                                file.write(('%d,' + '%g,' * 4 + '%s,%g\n') % (cap.get(cv2.CAP_PROP_POS_FRAMES), *xyxy, names[int(cls)], conf))

            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        df = pd.read_csv(output_path + name + '_FrameLevel_coco_v2.csv')
        
        df['area']=(df['x2']-df['x1'])*(df['y2']-df['y1'])/width/height
        with open(output_path + name + '_VideoLevel_coco_v2.csv', 'w') as file:
            file.write('Human Area Coverage\n')
            file.write(str(df[df['object']=='person']['area'].sum()/int(frames/skip_frame)))
        return True

    except:
       return  sys.exc_info()[1]

#get blurriness of grayscaled image
def image_blurriness(image):
    #convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #get variance of Laplacian as measure for blurriness
    return cv2.Laplacian(image, cv2.CV_64F).var()

#get colorfulness from image in RGB, frame needs to be passed
def image_colorfulness(image):
    # using Hasler, David ; Süsstrunk, Sabine, 2003: Measuring colourfulness in natural images
    
    # split the image into its respective RGB components
    R=image[:,0]
    G=image[:,1]
    B=image[:,2]
 
    # compute rg = R - G
    rg = R - G
 
    # compute yb = 0.5 * (R + G) - B
    yb = 0.5 * (R + G) - B
 
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
    # derive the "colorfulness" metric and return it
    return (stdRoot + (0.3 * meanRoot))/255

#get audio volume and ZCR
def audio_features(input_path, output_path, name, audio_folder):
    try:
        
        # create separate wav file
        clip = mp.VideoFileClip(input_path)
        clip.audio.write_audiofile(audio_folder+name+'_audio.wav')
        
        # get audio volume
        cut = lambda i: clip.audio.subclip(i,i+1).to_soundarray()
        volume = lambda array: np.sqrt(((1.0*array)**2).mean())
        volumes = [volume(cut(i)) for i in range(0,int(clip.audio.duration-2))] 
        try: clip.reader.close()
        except: pass
        try: clip.audio.reader.close_proc()
        except: pass
        write_csv(pd.DataFrame(volumes, columns =['Volume/s']),output_path,name+'_FrameLevel_audio_volume.csv')
        
        y, sr = librosa.load(audio_folder+name+'_audio.wav')
        zcr = pd.DataFrame(librosa.feature.zero_crossing_rate(y).transpose(), columns=['ZCR'])
        
        write_csv(zcr, output_path, name+'_FrameLevel_audio_zcr.csv')

        return True
    except: 
        sys.exc_info()[1]

#get an image in HSV space , frame needs to be passed
def HSVmetrics(hsvim):
    
    #hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    width, height, channels = hsvim.shape
    #print(hsvim.shape)
    
    #get Hh
    H=hsvim[:,:,0]
    A=np.sum(np.cos(np.pi*2*H/180))
    B=np.sum(np.sin(np.pi*2*H/180))
    Hh= np.arctan(B/A)/np.pi
    if A>0 and B<0:
        Hh += 2
    elif A<0:
        Hh += 1
    
    #get V
    V=1-(np.sqrt(A**2+B**2)/height/width)
    
    #get Hs
    S=hsvim[:,:,1]
    As=np.sum(S*np.cos(np.pi*2*H/180))/255
    Bs=np.sum(S*np.sin(np.pi*2*H/180))/255
    if As == 0:
        Hs=0
    else:
        Hs= np.arctan(Bs/As)/np.pi
    if As>0 and Bs<0:
        Hs += 2
    elif As<0:
        Hs += 1
        
    # get meanS, stdS, warmth, saturation_scaled_warmth
    meanS = np.mean(S)
    stdS = np.std(S)
    warmth=A/width/height
    saturation_scaled_warmth=As/width/height
    #Rn= np.sqrt(As**2+Bs**2)/len(S)**2
    
    #get Rs
    sumS=np.sum(S)
    if sumS == 0:
        Rs=0
    else:
        Rs= np.sqrt(As**2+Bs**2)/np.sum(S)*255
    
    # get meanV, stdV
    Value=hsvim[:,:,2]
    meanV=np.mean(Value)/255
    stdV=np.std(Value)/255
    
    # get pleasure, arousal, dominance
    pleasure = np.mean(0.69*Value[:]+0.22*S[:])
    arousal = np.mean(0.31*Value[:]+0.60*S[:])
    dominance = np.mean(-0.76*Value[:]+0.32*S[:])
    
    return Hh/2, V, warmth, saturation_scaled_warmth, meanS/255, stdS/255, Hs, Rs, meanV, stdV, pleasure, arousal, dominance

#identify colors defined by additional w2c file, frame needs to be passed
def image2colors(w2c, img_vec):
    img_vec=np.reshape(img_vec, (-1, 3))
    colors=w2c[np.array(np.floor(img_vec[:,0]/8)+32*np.floor(img_vec[:,1]/8)+1024*np.floor(img_vec[:,2]/8)).astype(int)]
    unique, counts = np.unique(colors, return_counts=True)
    counts = counts/len(img_vec)
    return unique, counts

#get colors dataframe defined by additional w2c file, frame needs to be passed
def get_color_share(w2c,img_vec):
    unique, counts = image2colors(w2c, img_vec)
    
    color=[]
    for i in range(1,12):
        if counts[np.where(unique == str(i))[0]]:
            color.append(round(np.array(counts[np.where(unique == str(i))])[0],3))
        else:
            color.append(0)
    return color

#get basic color stats incl RGB, brightness and colorfulness, frame needs to be passed
def basic_cols(hsv_frame):
    
    saturation = np.rint(np.average(hsv_frame[:,:,1]))/255
    value = np.rint(np.average(hsv_frame[:,:,2]))/255
    
    return saturation, value
