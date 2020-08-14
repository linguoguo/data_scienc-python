__author__ = 'lantos'

import unittest
import os
import Verification_Compare_Folder
import Merger
liste_of_tests = ["AR_1", "AR_2", "AR_3", "Enonce",
                  "FM_1", "FM_2", "FM_3",
                  "F_1", "F_2", "F_3",
                  "GM_1", "GM_2", "GM_3",
                  "P_1"]


class Test_Merger(unittest.TestCase):
    def test_Enonce(self):
        test = "Enonce"
        path_A = "Tests_Data" + os.sep + test + os.sep + "A"
        path_B = "Tests_Data" + os.sep + test + os.sep + "B"
        path_New = "Tests_Data" + os.sep + test + os.sep + "Save"

        Merger.Merge(path_A, path_B, path_New)

        path_reference = "Reference" + os.sep + test+ os.sep+"Save"

        self.assertTrue(Verification_Compare_Folder.compare_Folder(path_New, path_reference))
	
    def test_All_cases(self):
        isCorrect = True
        nbTests = len(liste_of_tests)
        nbPbs = 0
        for test in liste_of_tests:
            print "\n", test
            path_A = "Tests_Data" + os.sep + test + os.sep + "A"
            path_B = "Tests_Data" + os.sep + test + os.sep + "B"
            path_New = "Tests_Data" + os.sep + test + os.sep + "Save"
            path_reference = "Reference" + os.sep + test + os.sep + "Save"
            #if os.path.exists(path_New):
            #   os.system("rm -rf " + path_New)

            Merger.Merge(path_A, path_B, path_New)
            isIdentical1 = Verification_Compare_Folder.compare_Folder(path_New, path_reference)
            isIdentical2 = Verification_Compare_Folder.compare_Folder(path_reference, path_New)
            if not (isIdentical1 and isIdentical2):
                print "Error in case: " + test
                isCorrect = False
                nbPbs += 1
            else:
                print "OK"
            #os.system("rm -R " + path_New)
        print "Taux de reussite: ", float(nbTests - nbPbs) / float(nbTests)
        print "nb of Tests: ", nbTests, " nb of problems: ", nbPbs
        self.assertTrue(isCorrect)

if __name__ == '__main__':
    unittest.main()
