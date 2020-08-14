import os
import sys
import shutil
import stat
import string
from string import *
import Dossier as D
import manipulation_dossier as M



def Merge(Folder_1,Folder_2,New_folder):

	Folder_1_name=Folder_1.split('/')[-1]
	Folder_2_name=Folder_2.split('/')[-1]
	New_folder_name=New_folder.split('/')[-1]
	WF='/'.join(New_folder.split('/')[:-1])

	#	-------------------------  initialisation ----------------------------------


	if(os.path.getatime(WF+'/'+Folder_1_name) > os.path.getatime(WF+'/'+Folder_2_name)):
		B=D.Dossier(Folder_2_name,WF)
		A=D.Dossier(Folder_1_name,WF)
	else:
		B=D.Dossier(Folder_1_name,WF)
		A=D.Dossier(Folder_2_name,WF)
	Folder_1_name=A.getName()
	Folder_2_name=B.getName()

	""" ************* partie de repertoire ***************************"""
	#	-------------------------  creat a new folder --------------------------------
	A.new(New_folder_name)
	F_new=D.Dossier(New_folder_name,WF)

	#	-------------------------  depth of a folder ----------------------------------
	d1=M.L_lv(Folder_1_name,WF)[0]
	d2=M.L_lv(Folder_2_name,WF)[0]
	max_depth=max(d1,d2)


	#	--------------------  what's in folder 1 and 2? -----------------------
	A_nv_1=A.folder_dir(Folder_1_name)
	B_nv_1=B.folder_dir(Folder_2_name)



	#	-------------------------  copy of folders in depth with respectiv path ----------------------------------

	L_new=max_depth*[0]

	A_nv=A_nv_1
	B_nv=B_nv_1

	A_info=M.L_lv(Folder_1_name,WF)
	B_info=M.L_lv(Folder_2_name,WF)

	#	
	#	-------------------------  creation of doublon_lists ----------------------------------
	#	

	doublon_dpath_root = []
	doublon_folder = []
	doublon_dpath_depth = []

	for i in range(len(A_info[1])) :
		for j in range(len(A_info[1][i])) :
			for jj in A_info[1][i][j] :
				for k in B_nv_1 :
					if (jj == k) :
						doublon_dpath_root.append(B.path)
						doublon_folder.append(jj)
						doublon_dpath_depth.append(A_info[2][i][j])


	for i in range(len(B_info[1])) :
	    for j in range(len(B_info[1][i])) :
			for jj in B_info[1][i][j] :
				for k in A_nv_1 :
					if (jj == k) :
						doublon_dpath_depth.append(B_info[2][i][j])
						doublon_folder.append(jj)
						doublon_dpath_root.append(A.path)


	#	
	#	-------------------------  copy of directories ----------------------------------
	#	

	liste_file = []
	doublon_file_path = []
	doublon_file = []
	ok = 1

	#	
	#	-----------------------copy of the entire directories without duplicate at root------------------------------
	#	-----------------------and get the list of the files that we have copied------------------------------
	#	
	#	
	#	-----------------------copy of file contained in directories at root------------------------------
	#	

	for i in A_nv_1 :
	    if not (i in doublon_folder) :
	        os.system('cp -r '+WF+'/'+Folder_1_name+'/'+i+' '+WF+'/'+New_folder_name)
	        temp = D.Dossier(i,A.path)
	        if (temp.file_dir(i)) :
	            for j in temp.file_dir(i) :
	                doublon_file.append(j)
	                doublon_file_path.append(temp.path)

	for i in B_nv_1 :
	    if not (i in doublon_folder) :
	        os.system('cp -r '+WF+'/'+Folder_2_name+'/'+i+' '+WF+'/'+New_folder_name)
	        temp = D.Dossier(i,B.path)
	        if (temp.file_dir(i)) :
	            for j in temp.file_dir(i) :
	                for k in xrange(len(doublon_file)) :
	                    if(doublon_file[k] == j) :
	                        file2del_path = M.file_fusion(doublon_file_path[k]+'/'+j,temp.path+'/'+j,F_new.path+'/'+i)
	                        pathadd = M.niveau(j,file2del_path,WF)
	                        if (os.path.exists(F_new.path+'/'+pathadd[1]+'/'+j)) :
	                            os.system('rm '+F_new.path+'/'+pathadd[1]+'/'+j)
	                        ok=0
	                if(ok == 1) :
	                    doublon_file.append(j)
	                    doublon_file_path.append(temp.path)
	                ok = 1

	for i in xrange(len(doublon_folder)) :
		pathadd = M.niveau(doublon_folder[0],doublon_dpath_depth[0],WF)
		M.folder_fusion(doublon_dpath_root[i]+'/'+doublon_folder[i],doublon_dpath_depth[i]+'/'+doublon_folder[i],F_new.path+'/'+pathadd[1])

	for i in A.file_dir(Folder_1_name) :
	    if(os.path.exists(WF+'/'+Folder_2_name+'/'+i)) :
	        file2del_path = M.file_fusion(A.path+'/'+i,B.path+'/'+i,F_new.path)
	        pathadd = M.niveau(i,file2del_path,WF)
	        if (os.path.exists(F_new.path+'/'+pathadd[1]+'/'+i)) :
                        os.system('rm '+F_new.path+'/'+pathadd[1]+'/'+i)
	    elif(i in doublon_file) :
	        pathadd = M.niveau(i,doublon_file_path[doublon_file.index(i)],WF)
	        file2del_path = M.file_fusion(A.path+'/'+i,doublon_file_path[doublon_file.index(i)]+'/'+i,F_new.path+'/'+pathadd[1])
	        pathadd2 = M.niveau(i,file2del_path,WF)
	        if (os.path.exists(F_new.path+'/'+pathadd2[1])) :
                        os.system('rm '+F_new.path+'/'+pathadd2[1])
	    else :
	        os.system('cp '+A.path+'/'+i+' '+F_new.path)
	        doublon_file.append(i)
	        doublon_file_path.append(A.path)

	for i in B.file_dir(Folder_2_name) :
	    if not (os.path.exists(WF+'/'+Folder_1_name+'/'+i)) :
	        if(i in doublon_file) :
                    aa = M.niveau(i,B.path+'/'+i,WF)
                    bb = M.niveau(i,doublon_file_path[doublon_file.index(i)],WF)
                    if( aa[0] > bb[0]):
                        pathadd = aa[1]
                    else :
                        pathadd = bb[1]
	            file2del_path = M.file_fusion(B.path+'/'+i,doublon_file_path[doublon_file.index(i)]+'/'+i,F_new.path+'/'+pathadd)
	            pathadd2 = M.niveau(j,file2del_path,WF)
	            if (os.path.exists(F_new.path+'/'+pathadd2[1])) :
                        os.system('rm '+F_new.path+'/'+pathadd2[1])
	        else :
	            os.system('cp '+B.path+'/'+i+' '+F_new.path)
	            doublon_file.append(i)
	            doublon_file_path.append(B.path)

	#	
	#	-----------------------copy of file contained in depth------------------------------
	#	

	for i in range(A_info[0]) :
	    for j in range(len(A_info[1][i])) :
	        for k in A_info[1][i][j] :
	            temp = D.Dossier(k,A_info[2][i][j])
	            if (temp.file_dir(k)) :
	                for l in temp.file_dir(k) :
	                    for n in xrange(len(doublon_file)) :
	                        if (doublon_file[n] == l) :
	                            pathadd = M.niveau(doublon_file[n],temp.path,WF)
	                            file2del_path = M.file_fusion(doublon_file_path[n]+'/'+doublon_file[n],temp.path+'/'+l,F_new.path+'/'+pathadd[1])
	                            pathadd2 = M.niveau(l,file2del_path,WF)
	                            if (os.path.exists(F_new.path+'/'+pathadd2[1])) :
	                                os.system('rm '+F_new.path+'/'+pathadd2[1])
	                        if not (l in doublon_file) :
	                        	doublon_file.append(l)
	                        	doublon_file_path.append(temp.path)

	for i in range(B_info[0]) :
	    for j in range(len(B_info[1][i])) :
	        for k in B_info[1][i][j] :
	            temp = D.Dossier(k,B_info[2][i][j])
	            if (temp.file_dir(k)) :
	                for l in temp.file_dir(k) :
	                    for n in xrange(len(doublon_file)) :
	                        if (doublon_file[n] == l) :
	                            pathadd = M.niveau(doublon_file[n],temp.path,WF)
	                            file2del_path = M.file_fusion(doublon_file_path[n]+'/'+doublon_file[n],temp.path+'/'+l,F_new.path+'/'+pathadd[1])
	                            pathadd2 = M.niveau(l,file2del_path,WF)
	                            if (os.path.exists(F_new.path+'/'+pathadd2[1]+'/'+l)) :
	                                os.system('rm '+F_new.path+'/'+pathadd2[1]+'/'+l)
	                        if not (l in doublon_file) :
	                        	doublon_file.append(l)
	                        	doublon_file_path.append(temp.path)


