"""
objet Dossier
"""
import sys,os
import shutil

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def listfolderdir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
    		if not (x.startswith('.')) and os.path.isdir(directory+'/'+x)]

def listfiledir(directory):
    filelist = os.listdir(directory)
    return [x for x in filelist
    		if not (x.startswith('.')) and not os.path.isdir(directory+'/'+x)]
                        
class Dossier:
    def __init__(self,name,workFolder):
        """-------------  name = the name of the folder  ------------"""
        self._name = name
        """------------- workFolder= the path of the folder : name----------"""
        self._workFolder=workFolder
        """------------- path= the path of the folder : name----------"""
        self.path=self._workFolder+'/'+self._name
        
    def getName(self):
        return self._name
        
    def getWF(self):
    	return self._workFolder    
    
    """--------------- is Dossier(name) a folder ? -------------------"""
    def isDirectory(self):
    	return os.path.isdir(self.path)
    
    """---------------  creat a new folder: folder_name  --------------"""	
    def new(self,folder_name):
	if os.path.exists(self._workFolder + '/'+folder_name+'/'):
		shutil.rmtree(self._workFolder + '/'+folder_name+'/')
	os.mkdir(self._workFolder + '/'+folder_name+'/')
   	
    
    """----------------  creat a folder A in the folder B.path  ---------"""
    def creatAinB(self,A,B):
	if os.path.isdir(B) and  os.path.exists(B) and not os.path.exists(B+'/'+A+'/'):
	    #os.mkdir(self.path +'/'+A+'/')
	    os.mkdir(B+'/'+A+'/')

    """---------------------path split of the folder : name ----------------"""
    def path_split(self):
		return (self.path[1:]).split('/')	
	
	# ---------------------path split of the work foler ------------------ 
    def workFolder_split(self):
        return (self._workFolder[1:]).split('/')
        
    """---------------------- elements dans un repertoire-----------------"""
    def ele_dir(self,folder_name):
	if os.path.exists(self._workFolder + '/'+folder_name+'/'):
	    return mylistdir(self._workFolder+ '/'+folder_name)
	else:
	    print "the folder doesn't exists"        
        
    """---------------------- folders in the directory-----------------"""
    def folder_dir(self,folder_name):
	if os.path.exists(self._workFolder + '/'+folder_name+'/'):
	    return listfolderdir(self._workFolder+ '/'+folder_name)
	else:
	    print "the folder doesn't exists"  
    """---------------------- files in the directory-----------------"""
    def file_dir(self,folder_name):
	if os.path.exists(self._workFolder + '/'+folder_name+'/'):
	    return listfiledir(self._workFolder+ '/'+folder_name)
	else:
	    print "the file doesn't exists"  
	    
