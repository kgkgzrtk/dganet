from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from ctypes import *

import sys
import cv2
import numpy as np
import Image


IMAGE_W = 640
IMAGE_H = 480

area = 8
biasZ = 100

buf = None
tex_id = None

I_MAX = IMAGE_W/area -1 -1
J_MAX = IMAGE_H/area -1 -1


movX = 0
movY = 0
movZ = -500
angX = 0
angY = 0
movef = GL_FALSE
vertex = []
texture = []


dep_dir = "database/display/" + "depth757" + ".png"
dep_img = cv2.imread(dep_dir, cv2.IMREAD_GRAYSCALE)
dep_img = cv2.resize(dep_img, (IMAGE_W, IMAGE_H))
dep_img = (150.0 - dep_img.astype(np.float32))*3.0

col_dir = "database/display/" + "image" + ".png"
pcol = Image.open(col_dir)
pcol_img = np.array(list(pcol.getdata()),np.int8)


def display():

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glTranslatef(movX,movY,movZ)
    glRotate(angX, 1., 0 ,0)
    glRotate(angY, 0, 1., 0)
    print("rotated %f %f"%(angX, angY))

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    
    glBindBuffer(GL_ARRAY_BUFFER, buf[0])
    glTexCoordPointer(2, GL_FLOAT, 0, None)

    glBindBuffer(GL_ARRAY_BUFFER, buf[1])
    glVertexPointer(3, GL_FLOAT, 0, None)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D,tex_id)
    glDrawArrays(GL_QUADS, 0, len(vertex))
    glDisable(GL_TEXTURE_2D)

    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)
    glutSwapBuffers()

def spin_key(key, x, y):
    global angX, angY
    if key == GLUT_KEY_LEFT:
        angY -= 5.
    if key == GLUT_KEY_RIGHT:
        angY += 5.
    if key== GLUT_KEY_UP:
        angX -= 5.
    if key== GLUT_KEY_DOWN:
        angX += 5.
    glutPostRedisplay()


def keyboard(key, x, y):
    global movX, movY, movZ
    if key == 'a':
        movX += 5.
    if key == 'd':
        movX -= 5.
    if key == 'w':
        movZ += 5.
    if key == 's':
        movZ -= 5.
    if key == 'r':
        movY += 5.
    if key == 'f':
        movY -= 5.
    glutPostRedisplay()

def init_vertex(ver, tex):

    entX = float(area)/float(IMAGE_W)
    entY = float(area)/float(IMAGE_H)
    for _i in range(I_MAX):
        for _j in range(J_MAX):
            i=area*_i
            j=area*_j
            texX = float(i)/float(IMAGE_W)
            texY = float(j)/float(IMAGE_H)

            ver.extend([i-IMAGE_W/2, IMAGE_H-j-IMAGE_H/2, dep_img[j][i]-biasZ])
            ver.extend([i-IMAGE_W/2, IMAGE_H-(j+area)-IMAGE_H/2, dep_img[j+area][i]-biasZ])
            ver.extend([i+area-IMAGE_W/2, IMAGE_H-(j+area)-IMAGE_H/2, dep_img[j+area][i+area]-biasZ])
            ver.extend([i+area-IMAGE_W/2, IMAGE_H-j-IMAGE_H/2, dep_img[j][i+area]-biasZ])

            tex.extend([texX,texY])
            tex.extend([texX,texY+entY])
            tex.extend([texX+entX,texY+entY])
            tex.extend([texX+entX,texY])

def init_texture():
    global tex_id
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D,tex_id)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                    IMAGE_W,
                    IMAGE_H, 0,
                    GL_RGB,GL_UNSIGNED_BYTE,
                    pcol_img)
    glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE, GL_DECAL)
    glEnable(GL_AUTO_NORMAL)
    glEnable(GL_DEPTH_TEST)
    glFrontFace(GL_CCW)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

def init():
    global buf
    # select clearing color
    glClearColor(1.0, 1.0, 1.0, 0.0)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    buf = glGenBuffers(2)
    glBindBuffer(GL_ARRAY_BUFFER,buf[0])
    glBufferData(GL_ARRAY_BUFFER, len(texture)*4, (ctypes.c_float*len(texture))(*texture), GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER,buf[1])
    glBufferData(GL_ARRAY_BUFFER, len(vertex)*4, (ctypes.c_float*len(vertex))(*vertex), GL_STATIC_DRAW)


    # initialize viewing values
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(450.0, IMAGE_W/IMAGE_H, 0.1, 10000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_DEPTH_TEST)


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB)
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_REPEAT)
    glutInitWindowSize(IMAGE_W, IMAGE_H)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("3DVIEW")
    init_vertex(vertex,texture)
    init()
    init_texture()
    glutSpecialFunc(spin_key)
    glutKeyboardFunc(keyboard)
    glutDisplayFunc(display)
    print("Mainloop")
    glutMainLoop()

if __name__ == "__main__":
    main()
