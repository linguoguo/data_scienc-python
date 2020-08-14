__author__ = 'lantos'
import os
import hashlib


def compare_Folder(path_A, path_B):
    # print "\n Compare"
    isIdentical = True
    if not (os.path.isdir(path_A)):
        raise Exception("Folder\t" + path_A + " does not exist ")

    if not (os.path.isdir(path_B)):
        raise Exception("Folder\t" + path_B + " does not exist ")

    for root, dirs, files in os.walk(path_A, followlinks=True):
        # print "root A:", root
        tmp = os.sep.join(root.split(os.sep)[len(path_A.split(os.sep)):])
        # print "tmp=", tmp
        if tmp:
            tmp = tmp + os.sep
        for d in dirs:
            #print d
            #print path_B+os.sep+tmp+d
            if not os.path.isdir(path_B + os.sep + tmp + d):
                print "folder A = ", root + os.sep + d, " folder B does not exist= ", path_B + os.sep + tmp + d
                isIdentical = False

        for f in files:
            #print f
            #print path_B+os.sep+tmp+f
            if not f.startswith("."):
                if os.path.isfile(path_B + os.sep + tmp + f):

                    md5Sum_A = hashlib.md5(open(root + os.sep + f, 'r').read()).hexdigest()
                    md5Sum_B = hashlib.md5(open(path_B + os.sep + tmp + f, 'r').read()).hexdigest()
                    #isIdentical = (md5Sum_A == md5Sum_B)
                    if (md5Sum_A != md5Sum_B):
                        print "file A = ", root + os.sep + f, " file B does not exist= ", path_B + os.sep + tmp + f
                        return False
                else:
                    print "file A = ", root + os.sep + f, " file B does not exist= ", path_B + os.sep + tmp + f
                    return False

    return isIdentical