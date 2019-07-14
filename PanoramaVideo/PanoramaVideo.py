import numpy as np
import cv2
import sys
from functions import * 
import os


hp=os.path.abspath('__file__')
hp=hp.replace("\\","/")
hp=hp.replace('__file__','')
hp=hp+'PanoramaVideo/'

endframe=52*60*24+28*24;
beginframe=52*60*24+22*24-100;
#extract_frames('video.mp4', beginframe,endframe)

    #play_frames_kp('video.mp4',74942,80000,20000,0.25)

   


files=os.listdir('frames2')
ima=cv2.imread('frames2/'+files[0])
scale = 0.1
dim=ima.shape
width=dim[1];
height=dim[0];
draw=False 

ima=cv2.resize(ima,(int(scale*width),int(scale*height)))
for it in range(1,len(files)-1):
   

    imb=cv2.imread('frames2/'+files[it])

    print(files[it])

    ima=cv2.cvtColor(ima,cv2.COLOR_RGB2RGBA)

    imb=cv2.cvtColor(imb,cv2.COLOR_RGB2RGBA)


    imb=cv2.resize(imb,(int(scale*width),int(scale*height)))

    ima_gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY);
    imb_gray=cv2.cvtColor(imb,cv2.COLOR_BGR2GRAY);


    detector = cv2.xfeatures2d_SIFT.create()
    matcher = cv2.BFMatcher()


    kp_a, des_a=detector.detectAndCompute(ima_gray,None)
    kp_b, des_b=detector.detectAndCompute(imb_gray,None)

    matches = matcher.knnMatch(des_a,des_b, k=2)


    good=[]
    for m,n in matches:
        if m.distance <0.75*n.distance:
            good.append([m])

    src_pts=[]
    dst_pts=[]

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       flags = 2)

    img3=cv2.drawMatchesKnn(ima,kp_a,imb,kp_b,good,None,**draw_params)
    #cv2.imshow('img3',img3)

    src_pts = np.float32([ kp_a[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_b[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)



    M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5.0)

    M2, mask2 = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC,5.0)
    #Mi=np.linalg.inv(M);

    h,w,z = ima.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    ima_gray = cv2.polylines(imb_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    #cv2.imshow(' original image overlapping' ,ima_gray)

    dst=np.zeros((ima.shape[0]+imb.shape[0],ima.shape[1]+imb.shape[1],ima.shape[2]),np.uint8)
    starty=0
    startx=int(dst.shape[0]/2)

    #dst[starty:ima.shape[0]+starty,startx:ima.shape[1]+startx,:]=ima

    corner1=np.asarray([[0],[0],[1]]);
    corner2=np.asarray([[imb.shape[1]],[0],[1]])
    corner3=np.asarray([[0],[imb.shape[0]],[1]])
    corner4=np.asarray([[imb.shape[1]],[imb.shape[0]],[1]])
    def get_transformcoord(M,corner):
        corner_t=np.matmul(M,corner)
        return corner_t

    corner1_t=get_transformcoord(M,corner1)
    corner2_t=get_transformcoord(M,corner2)
    corner3_t=get_transformcoord(M,corner3)
    corner4_t=get_transformcoord(M,corner4)

    xnew=np.zeros((1,imb.shape[0]*imb.shape[1]))
    ynew=np.zeros((1,imb.shape[0]*imb.shape[1]))
    xold=np.zeros((1,imb.shape[0]*imb.shape[1]))
    yold=np.zeros((1,imb.shape[0]*imb.shape[1]))

    k=0

    #for i in range(imb.shape[0]):
    #    for j in range(imb.shape[1]):
    #       newxyz=np.matmul(M,[[j],[i],[1]])+np.asarray([[startx],[starty],[0]])
    #       xnew[0,k]=newxyz[0,0]
    #       ynew[0,k]=newxyz[1,0]
    #       xold[0,k]=j;
    #       yold[0,k]=i;
    #       k+=1
    #d=2;
    #xnew=np.asarray(xnew,int)[0]
    #ynew=np.asarray(ynew,int)[0]
    #xold=np.asarray(xold,int)[0]
    #yold=np.asarray(yold,int)[0]

    #dst[ynew,xnew,:]=imb[yold,xold,:]

    dst_b = cv2.warpPerspective(imb,M,(ima.shape[1] + imb.shape[1], ima.shape[0]+imb.shape[1]))
    dst_a=np.zeros(dst_b.shape,dtype=dst_b.dtype)
    dst_a[0:ima.shape[0],0:ima.shape[1]]=ima;
    mask_a=dst_a[:,:,3];
    mask_b=dst_b[:,:,3];
    mask_a=mask_a-mask_b;
    dst_a[:,:,3]=mask_a;


    dst_a[:,:,0]=np.multiply(dst_a[:,:,0],dst_a[:,:,3]/255)
    dst_a[:,:,1]=np.multiply(dst_a[:,:,1],dst_a[:,:,3]/255)
    dst_a[:,:,2]=np.multiply(dst_a[:,:,2],dst_a[:,:,3]/255)
    #dst_b[0:ima.shape[0],0:ima.shape[1],:]=ima;
    dst_b=dst_a+dst_b

    #dst2 = cv2.warpPerspective(imb,np.linalg.inv(M),(ima.shape[1] + imb.shape[1], ima.shape[0]+imb.shape[1]))
    #dst[0:ima.shape[0],0:ima.shape[1],:]=ima;
             #dst2 = cv2.warpPerspective(ima,M2,)
    #dst[0:ima.shape[0],0:ima.shape[1],:]=ima;

    #cv2.imshow('dst',dst_b)
    #cv2.imshow('dsta',dst_a)

    #cv2.imshow('dst2',dst2)
    gray=cv2.cvtColor(dst_b,cv2.COLOR_BGRA2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)


    ima_masked=ima
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    crop=dst_b[y:y+h,x:x+w,:]
    #cv2.imshow('crop',crop)
    cv2.imwrite('output2/'+files[it],crop)
    #cv2.imshow('dst_b',dst_b)
    if files[it]=='frame_75063.jpg':
        draw=True
 

    if draw:
        cv2.imshow('ima',ima)
        cv2.imshow('imb',imb)
        draw_points_a=cv2.drawKeypoints(ima,kp_a,None)
        draw_points_b=cv2.drawKeypoints(ima,kp_b,None)
        draw_matches = cv2.drawMatchesKnn(ima,kp_a,imb,kp_b,good,None,**draw_params)
        cv2.imshow('matches',draw_matches)
        cv2.imshow('dstb',dst_b)
        cv2.imshow('crop',crop)
        cv2.waitKey()


    


    #cv2.imshow('ima2',draw_points_a)
    #cv2.imshow('imb2',draw_points_b)
    #cv2.imshow('matches',draw_matches)

    
    ima=crop



test=2;
cv2.destroyAllWindows()