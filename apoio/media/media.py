import cv2
import numpy as np



def correlacao(n=1):
    mask = np.ones((n,n), dtype=int)

    smask  = len(mask)*len(mask)
    center = n // 2
    print( mask )


    #n vizinhos
    for i in range(lin-n):
        for j in range(col-n):
            val = 0
            for k in range(n): #indice da mascara
                for l in range(n):
                    val = val + img.item(i+k, j+l)*mask[k][l]

            val = val // smask
            cp.itemset((i+center,j+center), val)

def correlacao2(n=1):

    center = n // 2
    print(center)
    
    #n vizinhos
    for i in range(center,lin-center):
        for j in range(center,col-center):
            val = np.mean(img[i-center:i+center,j-center:j+center], dtype=int)
            
            cp.itemset((i,j), val)




img = cv2.imread("saltandpeppernoise.jpg", 0)
cp  =  img.copy()

lin, col = img.shape

print(img.shape)

#1 vizinho
    # for i in range(1,lin-1):
    #     for j in range(1,col-1):
    #         val = 0

    #         val = val + img.item(i-1,j)
    #         val = val + img.item(i,j)
    #         val = val + img.item(i+1,j)
    #         val = val + img.item(i,j-1)
    #         val = val + img.item(i,j+1)

    #         val = val + img.item(i-1,j-1)
    #         val = val + img.item(i-1,j+1)
    #         val = val + img.item(i-1,j+1)
    #         val = val + img.item(i+1,j+1)

    #         val = val // 9

    #         cp.itemset((i,j), val)



#correlacao(3)
correlacao2(3)

cv2.imshow("Imagem original", img)
cv2.waitKey(0)
cv2.imshow("Imagem modificada", cp)
cv2.waitKey(0)

cv2.imshow("Imagem modificada 2", cv2.medianBlur(img, 3))
cv2.waitKey(0)


