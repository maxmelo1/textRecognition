#-*- coding: utf-8 -*- 
 
import cv2 
import numpy as np
import imutils
#import statistics


def returnList(contours):
    ret = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ret.append(y+h)
    return ret

#Hu Moments
#requires a single channel img
def findFeaturesVector(img):
    return cv2.HuMoments(cv2.moments(img.copy())).flatten()
    
    
    

rois  = []

orig = cv2.imread('out.jpg')


orig = imutils.rotate_bound(orig, 90)
cv2.imwrite('rotated.png',orig)


height, width, channels = orig.shape


copy = orig.copy()

img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)


print height, width, channels

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find contours in the thresholded image
cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


vet = returnList(cnts)

hist, bin_edges = np.histogram(vet, bins = range(height))
    
print "len:", len(hist), "len edges:", len(bin_edges)
#print ",hist:", hist, "edges:", bin_edges


for x in range(0,len(hist)-1):  
    if hist[x]:
        print "indice:", x, ", valor:", hist[x]
        
     

# loop over the contours
for c in cnts:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
 
    # draw the contour and center of the shape on the image
    cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
    cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
    #cv2.rectangle(img, ( int( M["m10"] ), int( M["m01"] ) ), (cX, cY), (255, 0, 0), 1 )
    cv2.putText(img, "center", (cX - 20, cY - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    
    x,y,w,h = cv2.boundingRect(c)
    
    rois.append( copy[y:y+h, x:x+w] )
    
    
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.circle(orig, (x, y+h), 7, (255, 255, 255), -1)
    
    print "x:", x , "y:", y+h
    
    
    #data = [1, 5.8, 6.1, 6, 5.7, 6, 6.3, 80, 6]
 
    #print "v", statistics.median(data)     # returns 6.666666666666667
    #print "dp:", statistics.mode(data)        # returns 2.581988897471611
    
    #print cY
 
    # show the image    
    
    '''
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

i = 0
for roi in rois:
    bg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    print findFeaturesVector(bg)
    
    cv2.imwrite("roi/img"+str(i)+".png", roi)
    i +=1


cv2.imshow('Pressione s para salvar',orig)
k = cv2.waitKey(0) & 0xFF
#print k

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('detected.png',orig)
    cv2.destroyAllWindows()


