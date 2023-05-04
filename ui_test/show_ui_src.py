import sys,os,copy
from PyQt5.QtWidgets import  QApplication,QWidget,QMainWindow,QFileDialog,QMessageBox
from main_2 import Ui_MainWindow
from PyQt5 import QtCore,QtGui,QtPositioning
from automatic_mask_generator_example import *
envpath = '/home/uto/anaconda3/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


def check_result(img,mask_info,save_name):
    import copy
    src=copy.deepcopy(img)
    ##draw mask
    mask=mask_info["segmentation"]
    res = cv2.bitwise_and(src, src, mask=mask.astype(np.uint8))

    ##draw box
    bbox=mask_info["bbox"]
    cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 0), 1)  ##画矩形
    # plt.imshow(res)
    # plt.show()
    cv2.imwrite(save_name,res)


class my_window(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(my_window,self).__init__()
        self.setupUi(self)
        self.initUi()
        self.filelist=[]
        self.savePath=None
        try:
            self.model=load_model(device="cpu")  ##"cuda"
        except Exception as e:
            self.model=load_model(device="cpu")


    def initUi(self):
        self.pushButton.clicked.connect(self.loadFile)
        self.pushButton_2.clicked.connect(self.loadDir)
        self.pushButton_3.clicked.connect(self.modelSeg)
        self.pushButton_4.clicked.connect(self.nextFile)
        self.pushButton_5.clicked.connect(self.tagLabel)
        self.pushButton_6.clicked.connect(self.saveResult)

    def showImg(self,img_path):
        pixmap = QtGui.QPixmap(img_path).scaled(self.label_2.width(), self.label_2.height(),
                                                        QtCore.Qt.IgnoreAspectRatio)
        self.label_2.setPixmap(pixmap)
        self.label_2.setScaledContents(True)

    def mergeArea(self,masks):
        # import pdb;pdb.set_trace()
        size=len(masks)
        res=copy.deepcopy(masks)
        try:
            for cnt1 in range(size):
                for cnt2 in range(cnt1+1,size):
                    tmp=np.logical_and(masks[cnt1]["segmentation"],masks[cnt2]["segmentation"])
                    if (tmp==masks[cnt1]["segmentation"]).all():
                        res[cnt1]=0
                    elif (tmp==masks[cnt2]["segmentation"]).all():
                        res[cnt2]=0
        except Exception as e:
            print(e)

        t=0
        while t<len(res):
            if res[t]==0:
                res.pop(t)
                t-=1
            t+=1

        return res


    def showMask(self,anns,src):
        do_mask=False
        try:
            img=copy.deepcopy(src)
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            # all_contours=[]
            for ann in sorted_anns:
                m = ann['segmentation']*255
                # _, binary = cv2.threshold(m, 0, 100, cv2.THRESH_BINARY)
                kernel = np.ones((1, 5), np.uint8)
                binary = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # all_contours.append(contours)
                color=np.random.randint(0, 255, 3, dtype=np.int32)
                cv2.drawContours(img, contours, -1, (int(color[0]), int(color[1]), int(color[2])), 3)

            convertToQtFormat = QtGui.QImage(img, img.shape[1], img.shape[0],QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(self.label_2.width(), self.label_2.height(), QtCore.Qt.IgnoreAspectRatio)
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(p))
            self.label_2.setScaledContents(True)
            do_mask=True
        except Exception as e:
            print("error occur:",e)
        return do_mask

    def loadFile(self):
        self.filelist.clear()
        fileName, fileType = QFileDialog.getOpenFileName(self,
                                                           "打开文件",
                                                           "",
                                                           "All Files (*)")
        if os.path.isfile(fileName) and fileName.split(".")[-1] in ["jpg","png","jpeg"]:
            self.filelist.append(fileName)
            self.savePath=os.path.basename(fileName)

        if(len(self.filelist)):
            self.showImg(self.filelist[0])

    def loadDir(self):
        self.filelist.clear()
        file_names=QFileDialog.getExistingDirectory(self,"选择文件夹",os.getcwd())
        if os.path.isdir(file_names):
            self.savePath=os.path.basename(file_names)+"/result"
            os.makedirs(self.savePath,exist_ok=True)
            for cnt,file in enumerate(sorted(os.listdir(file_names))):
                self.filelist.append(os.path.join(file_names,file))


        if(len(self.filelist)):
            self.showImg(self.filelist[0])

    def modelSeg(self):
        """
        ###当完成推理后，self.label_2显示推理后的图像，self.label_3实时显示鼠标在self.label＿２中选择的mask部分
        ###鼠标左键表示选中区域，右键表示回撤上一步的选择,当一切完成选择后，通过finish按钮给选择的部分打上一个标签，
        ###然后再将其掩膜save出来
        """
        if(len(self.filelist)):
            for index,file in enumerate(self.filelist):
                image = cv2.imread(file)
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                res_=model_seg2(self.model,img)
                res=self.mergeArea(res_)  ###去掉部分分割存在包含关系的
                do_seg=self.showMask(res,image)

                ###generate label
                ###to check mask
                # for cnt in range(len(res)):
                #     check_result(img,res[cnt],"images/t/"+str(cnt)+".jpg")

                self.label_mask=np.zeros((img.shape[0],img.shape[1]))
                for cnt in range(len(res)):
                    seg_lab=(res[cnt]["segmentation"]*(cnt+1)).astype(np.int8)
                    # np.savetxt("images/result/"+str(cnt+1)+".txt",seg_lab)
                    if np.sum(np.logical_and(self.label_mask,seg_lab))==0:
                        self.label_mask+=seg_lab
                    else:
                        print("error occur",cnt)
                # np.savetxt("images/result/all.txt",self.label_mask)

                import pdb;pdb.set_trace()
                if do_seg:
                    #todo mouse action
                    pass

        else:
            QMessageBox.warning(self,"警告","请先load图片",QMessageBox.Cancel)

    def nextFile(self):
        if(len(self.filelist)):
            self.filelist.pop(0)
            if(len(self.filelist)):
                self.showImg(self.filelist[0])
            else:
                QMessageBox.warning(self, "警告", "filelist is empty", QMessageBox.Cancel)
        else:
            QMessageBox.warning(self, "警告", "filelist is empty", QMessageBox.Cancel)

    def tagLabel(self):
        """
        　　获得此前通过鼠标选择的区域，然后通过选择对其跳出一个窗口，赋予一个标签
        """
        pass


    def saveResult(self):
        """
            对完成标签的mask进行保存
        """
        pass

 
if __name__=="__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  ###保证按原来布局正常显示
    app=QApplication(sys.argv)
    t=my_window()
    t.show()
    app.exec_()
