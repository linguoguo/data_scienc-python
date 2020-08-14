#! /usr/bin/python
import os
import sys
import shutil
import stat
import string
import time
from string import *
import Dossier as D


def L_lv(A,P_A):
	F_lv=[]
	P_lv=[]

	lv_1=D.Dossier(A,P_A)
	l=lv_1.folder_dir(A)
	l_lv_1=[]
	nb_d=len(l) #nb de dir
	for i in range(nb_d):
		l_lv_1.append([l[i]])
		path_lv_1=nb_d*[lv_1.path]
	k=0
	while len(l_lv_1)>0:
		nb_d=len(l_lv_1)
		l_lv_2=[]
		path_lv_2=[]
		for i in range(nb_d):
			nb_d_tmp=len(l_lv_1[i])
			for j in range(nb_d_tmp):
				dir=D.Dossier(l_lv_1[i][j],path_lv_1[i])
				if dir.folder_dir(dir.getName()):
					l_lv_2.append(dir.folder_dir(dir.getName()))
					path_lv_2.append(dir.path)

		k+=1
		l_lv_1=l_lv_2
		path_lv_1=path_lv_2
		F_lv.append(l_lv_1)
		P_lv.append(path_lv_1)

	return k,F_lv,P_lv

def elt_lv(A,P_A):
	F_lv=[]
	P_lv=[]

	lv_1=D.Dossier(A,P_A)
	l=lv_1.folder_dir(A)
	l_lv_1=[]
	nb_d=0
	if lv_1:
		nb_d=len(l) #nb de dir
	for i in range(nb_d):
		l_lv_1.append([l[i]])
		path_lv_1=nb_d*[lv_1.path]
	k=0
	while len(l_lv_1)>0:
		nb_d=len(l_lv_1)
		l_lv_2=[]
		path_lv_2=[]
		for i in range(nb_d):
			nb_d_tmp=len(l_lv_1[i])
			for j in range(nb_d_tmp):
				dir=D.Dossier(l_lv_1[i][j],path_lv_1[i])
				if dir.folder_dir(dir.getName()):
					l_lv_2.append(dir.ele_dir(dir.getName()))
					path_lv_2.append(dir.path)

		k+=1
		l_lv_1=l_lv_2
		path_lv_1=path_lv_2
		F_lv.append(l_lv_1)
		P_lv.append(path_lv_1)

	return k,F_lv,P_lv

def niveau(A,P,WF):
	dir=D.Dossier(A,P)
	dsp=dir.path_split()
	wsp=WF[1:].split('/')
	niv=len(dsp)-len(wsp)
	diff='/'.join(dsp[len(wsp)+1:])
	return niv,diff

def file_fusion(path_file_1,path_file_2,path_folder_new) :
    if(os.path.getatime(path_file_1) > os.path.getatime(path_file_2)) :
        os.system('cp '+path_file_1+' '+path_folder_new)
        print " "
        print "The file located at the following path : "
        print path_file_2
        print "and created on"
        print time.strftime("%d/%m/%Y-%H:%M:%S",time.gmtime(os.path.getatime(path_file_2)))
        print "has been replaced by the file located at the following path : "
        print path_file_1
        print "because it's been created on "
        print time.strftime("%d/%m/%Y-%H:%M:%S",time.gmtime(os.path.getatime(path_file_1)))
        print " "
        return '/'.join(path_file_2.split('/')[:-1])
    else :
        os.system('cp '+path_file_2+' '+path_folder_new)
        print " "
        print "The file located at the following path : "
        print path_file_1
        print "and created on"
        print time.strftime("%d/%m/%Y-%H:%M:%S",time.gmtime(os.path.getatime(path_file_1)))
        print "has been replaced by the file located at the following path : "
        print path_file_2
        print "because it's been created on "
        print time.strftime("%d/%m/%Y-%H:%M:%S",time.gmtime(os.path.getatime(path_file_2)))
        print " "
        return '/'.join(path_file_1.split('/')[:-1])

def folder_fusion(path_file_1,path_file_2,path_folder_new) :
    if(len(path_file_1) > len(path_file_2)) :
        if(os.path.exists(path_folder_new)) :
            shutil.rmtree(path_folder_new)
            os.system('cp -r '+path_file_1+'/* ' + path_folder_new)
            os.system('cp -r '+path_file_2+'/* ' + path_folder_new)
        else :
            os.system('cp -r '+path_file_1+' '+path_folder_new)
            os.system('cp -r '+path_file_2+'/* ' + path_folder_new)
        return [path_file_1]
    else :
        if(os.path.exists(path_folder_new)) :
            shutil.rmtree(path_folder_new)
            os.system('cp -r '+path_file_2+' '+path_folder_new)
            os.system('cp -r '+path_file_1+'/* '+path_folder_new)
        else :
            os.system('cp -r '+path_file_2+'/* '+path_folder_new)
            os.system('cp -r '+path_file_1+'/* '+path_folder_new)
        return [path_file_2]

"""---------------------- folders in the directory-----------------"""
def file_dir(A,P_A):
    a=D.Dossier(A,P_A)
    return a.file_dir
'''
def file_dir_fusion(D,path_ant,path_2,new_folder,WF):
	c_1=D.Dossier(D,path_1)
	c_2=D.Dossier(D,path_2)
	c_1_sp=c_1.path_split()
	c_2_sp=c_2_sp.path_split()
	wsp=WF[1:].split('/')
	niv1=len(c_1_sp)-len(wsp)
	niv2=len(c_2_sp)-len(wsp)
	if niv1<niv2:
		for i in c_1.file_dir:
			if i in
'''
