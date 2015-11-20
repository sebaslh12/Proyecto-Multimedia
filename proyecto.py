import sys
from Tkinter import *
import tkMessageBox
from PIL import Image
import numpy as np
from numpy import array
import cv2
import math

#obtener y tratar la imagen
def get_image(source,size=(128,128)):
    image = Image.open(source)    
    img_color = image.resize(size, 1)
    img_grey = img_color.convert('L')
    img = np.array(img_grey, dtype=np.float)
    img.tolist()
    return img
#Dimensiones de la region
N=M=8

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
#coeficientes
def coef(Fxy,u,v):
    coef = 0
    for x in range(0, N):
        for y in range(0, M):
            val_xy = Fxy[x][y]
            coef += val_xy * g(x,y,u,v)
            #print coef
    return coef
#g
def g(x,y,u,v):
   return alpha_u(u) * alpha_v(v) * math.cos((2*x+1)*u*math.pi/(2*N)) * math.cos((2*y+1)*v*math.pi/(2*M))
#alphau
def alpha_u(u):
    if u == 0:
        return math.sqrt(1.0/N)
    else:
        return math.sqrt(2.0/N)
#alphav
def alpha_v(v):
    if v == 0:
        return math.sqrt(1.0/M)
    else:
        return math.sqrt(2.0/M)
#funcion para cuantificar
def quantize(T, Q):
    for u in range(0, N):
        for v in range(0, M):
            T[u][v] /= Q[u][v] * 1.0
            T[u][v] = int(round(T[u][v]))
    return T

#funcion para decuantificar
def dequantize(T, Q):
    for u in range(0,N):
        for v in range(0,M):
            T[u][v]*=Q[u][v] * 1.0
            T[u][v] = int(round(T[u][v]))
    return T

#transformada inversa del coseno
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
#Funcion para cambiar de tipos, nparray to list
def arraytolist(F):    
    for i in range(0,len(F)):
        F[i]=F[i].tolist()
    return F        
#Funcion que realiza los bloques de arrays de len 8 horizontalmente
def divocho(F):
    Divisiones=[]    
    for i in range(0,len(F)):
        div=[]
        temp=[]
        F[i]=F[i].tolist()
        for j in range(0,len(F[0])):        
            if (j%8)!=0:
                temp.append(F[i][j])
            else:
                div.append(temp)
                temp=[]
                temp.append(F[i][j])                
        del div[0]
        div.append(temp)
        Divisiones.append(div)
    
    return Divisiones
#Funcion que divide por regiones de 8x8
def regiones(F,j):
    Divisiones=[]
    temp=[]    
    for i in range(0,len(F)):        
        if (i%8)!=0:
            temp.append(F[i][j])
        else:            
            Divisiones.append(temp)
            temp=[]
            temp.append(F[i][j])                
    del Divisiones[0]        
    Divisiones.append(temp)    
    return Divisiones
#funcion auxiliar para desregionalizar, obtener bloques 1x128
def desocho(F):
    RawImg=[]
    for i in range(0,16):
        tmp=[]
        for j in range(0,len(F[0])):            
            RawImg.append(F[i][j])
    return RawImg
#funcion auxiliar para desregionalizar, concatenar los bloques 1x128
def desreg(R1,R2):
    N=[]
    for i in range(0,len(R1)):
        N.append(R1[i]+R2[i])
    return N
#Funcion que devuelve la imagen a la normalidad
def desregion(Resultado):
    Salida=desocho(Resultado)
    del Resultado[0:16]
    for index in range(0,15):
        Temp=desocho(Resultado)
        Salida=desreg(Salida,Temp)
        del Resultado[0:16]        
    return Salida

#Funcion principal que realiza dct, idct, divide en regiones y deja el formato de la imagen original
def Proyecto(Archivo):
    R= get_image(Archivo)
    R=arraytolist(R)
    reg=divocho(R)
    Regiones=[]
    for i in range(0,len(reg[0])):
        Regiones=Regiones+(regiones(reg,i))        
    Resultado=[]
    for i in range(0,len(Regiones)):
        Fxy = shift(Regiones[i],8)    
        #dct
        Tuv = dct(Fxy)
        #cuantificacion
        Tuv = quantize(Tuv,Q)
        #descuantificacion
        decTuv = dequantize(Tuv,Q)
        #idct
        F2xy = idct(Tuv)
        #shift inv
        F2xy = shift_inv(F2xy,8)    
        Resultado.append(F2xy)
    ImagenFinal=desregion(Resultado)
    ImagenFinal=np.asarray(ImagenFinal)
    #Guardar imagen
    cv2.imwrite('Dct.jpg',ImagenFinal)
    return ImagenFinal

def GUICall1(Archivo):
    text=Archivo.get()    
    if len(text)>0:
        Img=Proyecto(text)
    else:
         tkMessageBox.showinfo("IDCT Python","Digite el nombre de la imagen")
    #CODIGO PARA GUARDAR LA IMAGEN QUE ESTA GUARDA EN Img

def GUICall2(Archivo):
    #Img=Proyecto(Archivo.get())
    text=Archivo.get()    
    if len(text)>0:        
        #CODIGO PARA GUARDAR LA IMAGEN QUE ESTA GUARDA EN Img
        #CODIGO PARA MOSTRAR LAS IMAGENES, LA QUE SE ABRE POR EL INPUT Y LA QUE SE GENERA
        cv2.imshow("Imagen",cv2.imread(Archivo.get()))
        cv2.imshow("Compresion",cv2.imread("Dct.jpg"))
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
    else:
         tkMessageBox.showinfo("IDCT Python","Digite el nombre de la imagen")
    
def main(window):    
    window.title("IDCT Python")    
    Frame1 = Frame(window)
    
    #Componentes definidos
    Label(Frame1, text="Nombre del archivo: ").grid(row=0)
    Entrada=Entry(Frame1)
    Para_resolver=Button(Frame1)
    Para_resolver["text"]="Comprimir"
    Para_resolver["width"]=15
    Validar=Button(Frame1)
    Validar["text"]="Mostrar Imagenes"
    Validar["width"]=15
    
    #Asignacion de comandos (funciones activadas al presionar, solo botones)
    Para_resolver["command"] =  lambda win=window, frame1=Frame1, input1=Entrada: GUICall1(input1)
    Validar["command"] = lambda win=window, frame1=Frame1, input1=Entrada: GUICall2(input1)

    #Ubicacion de los elementos en el grid
    Entrada.grid(row=0, column=1)
    Para_resolver.grid(row=0, column=2)
    Validar.grid(row=1,column=2)
    Frame1.grid(row = 0, column = 0)
#DCTRecursiva()
root = Tk()
main(window=root)
root.mainloop()

