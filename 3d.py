from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import sys
import cv2
import numpy as np

IMAGE_W = 640
IMAGE_H = 480

area = 4

dwnX = 0
dwnY = 0
angX = 0
angY = 0
movef = GL_FALSE

t_ = [320.0, 240.0, 300.0]
v_ = [640.0 + 400.0, 480.0 + 200.0, 1200.0]

col_dir = "database/display/" + "image" + ".png"
dep_dir = "database/display/" + "depth" + ".png"
col_img = cv2.imread(col_dir)
dep_img = cv2.imread(dep_dir, cv2.IMREAD_GRAYSCALE)
col_img = cv2.resize(col_img, (IMAGE_W, IMAGE_H))
dep_img = cv2.resize(dep_img, (IMAGE_W, IMAGE_H))

col_img = col_img.astype(np.float32)/255.0
dep_img = (150.0 - dep_img.astype(np.float32))*3.0

def degree(r):
    return r * 180. / np.pi

def putup(i,j):
    # draw white polygon
    glColor3f(col_img[j][i][2],col_img[j][i][1] , col_img[j][i][0])
    glVertex3f(i, IMAGE_H-j, dep_img[j][i])

    glColor3f(col_img[j+area][i][2],col_img[j+area][i][1] , col_img[j+area][i][0])
    glVertex3f(i, IMAGE_H-(j+area), dep_img[j+area][i])

    glColor3f(col_img[j+area][i+area][2],col_img[j+area][i+area][1] , col_img[j+area][i+area][0])
    glVertex3f(i+area, IMAGE_H-(j+area), dep_img[j+area][i+area])

    glColor3f(col_img[j][i+area][2],col_img[j][i+area][1] , col_img[j][i+area][0])
    glVertex3f(i+area, IMAGE_H-j, dep_img[j][i+area])

def display():
    global angX, angY
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()

    rotateX = np.cos(angX) * np.cos(angY)
    rotateY = np.sin(angY)
    rotateZ = np.sin(angX) * np.cos(angY)
    
    glTranslatef(t_[0],t_[1],t_[2])
    glRotate(degree(rotateX), 1, 0 ,0)
    glRotate(degree(rotateY), 0, 1, 0)
    glRotate(degree(rotateZ), 0, 0, 1)
    glTranslatef(-t_[0],-t_[1],-t_[2])

    print("rotated %f %f %f" % (rotateX,rotateY, rotateZ))

    glBegin(GL_QUADS)
    for i in range(IMAGE_W/area - 1 - 1):
        for j in range(IMAGE_H/area - 1 - 1):
            putup(i*area,j*area)
    glEnd()
    glFlush()
    glPopMatrix()


def mouse(button, state, x, y):
    global dwnX, dwnY, movef
    if button == GLUT_LEFT_BUTTON & state == GLUT_DOWN:
        print "click!"
        dwnX = x/32.
        dwnY = y/32.
        movef = GL_TRUE
    if button == GLUT_LEFT_BUTTON & state == GLUT_UP:
        movef = GL_FALSE

def drag(x,y):
    global angX, angY, dwnX, dwnY, movef
    if movef == GL_TRUE:
        angX += x/32. - dwnX
        angY += y/32. - dwnY
        dwnX = x/32.
        dwnY = y/32.
        glutPostRedisplay()


def init():
    # select clearing color
    glClearColor(1.0, 1.0, 1.0, 0.0)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    glFrontFace(GL_CCW)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # initialize viewing values
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, IMAGE_W/IMAGE_H, 0.1, 10000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    #v_ = [20.0, 20.0, 500.0]
    #t_ = [10.0, 10.0, 0.0]
    gluLookAt(v_[0], v_[1], v_[2], t_[0], t_[1], t_[2] , 0.0, 1.0, 0.0)

    #glOrtho(0, IMAGE_W, IMAGE_H, 0, 0, 255)


def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(IMAGE_W, IMAGE_H)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("3DVIEW")
    init()
    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(drag)
    glutSwapBuffers()
    glutMainLoop()

if __name__ == "__main__":
    main()
