import cv2
import numpy as np
import sys
sys.path.append('../deploy')
import os
import argparse
import retinaface_model




def load_model():
    parser = argparse.ArgumentParser(description='face model test')
        # general
    parser.add_argument('--model_path', default='../RetinaFace/mnet.25/mnet.25',
                            help='model_path for face detection')
    parser.add_argument('--model', default=00, type=int,
                            help='model.')
    parser.add_argument('--gpuid', default=0, type=int, help='gpu id')
    parser.add_argument('--net', default='net3',
                            help='net')

    parser.add_argument('--model_recog', default='../recognition/model/y2-arcface-retina/model,01',
                            help='path to load model for face recognition.')
    args = parser.parse_args()
    return retinaface_model.FaceModel(args)

def read_and_register(path, save_path, id_file, id_embedding):
    model = load_model()
    path = '/home/wangshanmin/academy/insightface/interface/实验室图片'
    filelist = os.listdir(path)
    fid_name = open(id_file, 'wt')
    fid_feature = open(id_embedding, 'wt')
    for file in filelist:
        name = file
        print(name)
        fid_name.write(name + '\n')
        file = os.path.join(path, file)
        imglist = os.listdir(file)
        for i in range(len(imglist)):
            img = os.path.join(file, imglist[i])
            image = cv2.imread(img)
            image = cv2.resize(image, (256, 256))
            ####导入模型， 提取特征
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            bbox, aligned_face = model.get_input(image)
            feature = model.get_feature(aligned_face[0])
            ###保存图片
            img_path_new = os.path.join(save_path, name)
            if not os.path.exists(img_path_new):
                os.makedirs(img_path_new)
            img_name_new = img_path_new + '/' + str(i) + '.jpg'
            print(img_name_new)
            cv2.imwrite(img_name_new, image)
            ###保存特征
            for emb in feature:
                fid_feature.write(str(emb) + ' ')
            fid_feature.write(name + '_' + str(i))
            fid_feature.write('\n')
    fid_name.close()
    fid_feature.close()


if __name__=='__main__':
    path = '/home/wangshanmin/academy/insightface/interface/实验室图片'
    save_path = './register_img/'
    id_file = 'id_file.txt'
    id_embedding = 'id_embedding.txt'
    read_and_register(path, save_path, id_file, id_embedding)