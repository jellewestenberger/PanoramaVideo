import cv2
 

def extract_frames(videofile,startframe,endframe,set):
    cap = cv2.VideoCapture(videofile)
    fps= cap.get(5)
    cap.set(1,startframe-1)
    while(cap.isOpened()):
   
        ret, frame = cap.read()
        nr=int(cap.get(1))

        bezel=200;
        width=frame.shape[1];
        height=frame.shape[0];
        
        frame=frame[bezel:height-bezel,:,:]
        cv2.imshow('frame',frame)
        a=cv2.imwrite('frames'+str(set)+'/frame_'+str(nr)+'.jpg',frame)
        if nr==endframe:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        dummy=2

    cap.release()
    cv2.destroyAllWindows()
    return 


def play_frames_kp(videofile,startframe,endframe,threshold,scale):
    cap = cv2.VideoCapture(videofile)
    det=cv2.xfeatures2d_SIFT.create()
    matcher = cv2.BFMatcher()


    fps= cap.get(5)
    cap.set(1,startframe)
    cap2=cap;
    cap2.set(1,startframe+20)
    while(cap.isOpened()):
   
        ret, frame = cap.read()
        ret2,frame2= cap2.read()  

        dim=frame.shape
        width=dim[1];
        height=dim[0];
    
        frame=cv2.resize(frame,(int(scale*width),int(scale*height)))
        frame2=cv2.resize(frame2,(int(scale*width),int(scale*height)))

        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame2_gray=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        kp,des=det.detectAndCompute(frame_gray,None)
        kp2,des2=det.detectAndCompute(frame2_gray,None)
        matches= matcher.knnMatch(des,des2, k=2)
        good=[]
        for m,n in matches:
            if m.distance <0.75*n.distance:
                good.append([m])

        cv2.drawKeypoints(frame,kp,frame)
        cv2.drawKeypoints(frame2,kp2,frame2)
        draw_match=cv2.drawMatchesKnn(frame,kp,frame2,kp2,good,None,flags=2)
       
        nr=int(cap2.get(1))
        


        #cv2.imshow('frame',frame)
        #cv2.imshow('frame2',frame2)
        cv2.imshow('matches',draw_match)
        if nr==endframe:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    return 

def crop_blacks(dst_b):
    gray=cv2.cvtColor(dst_b,cv2.COLOR_BGRA2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop=dst_b[y:y+h,x:x+w,:]
    return crop