from PIL import Image
import numpy as np
from numpy import array
import cv2
import math
#array para guardar la imagen de 512 x 512
darray = np.empty([2,3,3], dtype=np.uint8)
#cv2 obtiene la matriz double que corresponde a la imagen
darray[0] = cv2.imread("Cameramanpeque.tif", 0)
#Tamaño
print(darray[0].shape)
#Tipo
print(type(darray[0]))
#darray[0][pixel] 
#print(darray[0][511]) #Muestra la matriz que representa cada uno de los pixeles

#cv2.imshow("face", darray[0])
#cv2.waitKey(3000)
#Dimensiones
Fxy=darray[0]
M=len(darray[0])
N=len(darray[0][0])
print N , M
#Matriz de cuantificacion
Q = [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]]

def printBitmap(F, pdfFormat = False):
    if not pdfFormat:
        for x in range(0, len(F)):
            print F[x]
    else:
        for x in range(0, len(F)):
            line = ""
            for y in range(0, len(F[0])):
                if y == len(F[0]) - 1:
                    line += str(F[x][y]) + " \\\\ "
                else:
                    line += str(F[x][y]) + " & "
            print line
    print
        
def printList(l, pdfFormat = False):
    if not pdfFormat:
        print l
    else:
        line = ""
        for y in range(0, len(l)):
            if y == len(l) - 1:
                line += str(l[y]) + " \\\\ "
            else:
                line += str(l[y]) + " \,  "
        print line

    print
        
    
def shift(F,b):
    n = int(math.pow(2,b-1))

    for x in range(0, N):
        for y in range(0, M):
            F[x][y] -= n

    return F

def dct(Fxy):
    Tuv = list()
    for u in range(0, N):
        Tuv_row = list()
        for v in range(0, M):
            Tuv_row.append(int(round(coef(Fxy,u,v))))
        Tuv.append(Tuv_row)
    return Tuv

def coef(Fxy,u,v):
    coef = 0
    for x in range(0, N):
        for y in range(0, M):
            val_xy = Fxy[x][y]
            coef += val_xy * g(x,y,u,v)
            #print coef
    return coef

def g(x,y,u,v):
   return alpha_u(u) * alpha_v(v) * math.cos((2*x+1)*u*math.pi/(2*N)) * math.cos((2*y+1)*v*math.pi/(2*M))

def alpha_u(u):
    if u == 0:
        return math.sqrt(1.0/N)
    else:
        return math.sqrt(2.0/N)

def alpha_v(v):
    if v == 0:
        return math.sqrt(1.0/M)
    else:
        return math.sqrt(2.0/M)

def quantize(T, Q):
    for u in range(0, N):
        for v in range(0, M):
            T[u][v] /= Q[u][v] * 1.0
            T[u][v] = int(round(T[u][v]))
    return T

#Hacer funcion para decuantificar
def dequantize(T, Q):
    for u in range(0,N):
        for v in range(0,M):
            T[u][v]*=Q[u][v] * 1.0
            T[u][v] = int(round(T[u][v]))
    return T

#Hacer transformada inversa del coseno
def idct(T):
    F = list()
    for u in range(0, N):
        F_row = list()
        for v in range(0, M):
            F_row.append(int(round(icoef(T,u,v))))
        F.append(F_row)
    return F

#Funcion auxiliar para calcular para hacer operacioned de la idct
def icoef(T,x,y):
    icoef = 0
    for u in range(0, N):
        for v in range(0, M):
            val_xy = T[u][v]
            icoef += val_xy * g(x,y,u,v)            
            #print coef    
    return icoef

#Complemento a dos invertido
def shift_inv(F,b):
    n = int(math.pow(2,b-1))

    for x in range(0, N):
        for y in range(0, M):
            F[x][y] += n

    return F

#---------------------------------   

print "Imagen Inicial"
printBitmap(Fxy)

Fxy = shift(Fxy,8)
print "shift"
printBitmap(Fxy)

Tuv = dct(Fxy)
print "DCT"
printBitmap(Tuv)
print(type(Tuv))

print "Q"
printBitmap(Q)

Tuv = quantize(Tuv,Q)
print "Cuantificacion"
printBitmap(Tuv)

# Tuv_zig_zag = zig_zag(Tuv)
# print "ZIG-ZAG scan"
# printList(Tuv_zig_zag)

decTuv = dequantize(Tuv,Q)
print "Decuantificación"
printBitmap(decTuv)

F2xy = idct(Tuv)
print "IDCT"
printBitmap(F2xy)

F2xy = shift_inv(F2xy,8)
print "Shift inv"
printBitmap(F2xy)
cv2.imshow("2", darray[0])
cv2.imshow("face", np.asarray(F2xy))
cv2.waitKey(3000)



