import os
import numpy as np
import csv
def prediction(path_im, path_hr):
    frames_per_step = 50
    list_dir = sorted(os.listdir(path_im))

    #df = open('/media/bousefsa1/Elements/v4v_challenge/gt.txt', 'w')
    for i in range(int(len(list_dir))):
        list_dir_im = sorted(os.listdir(path_im + '/' + list_dir[i]))
        list_dir_hr = sorted(os.listdir(path_hr + '/' + list_dir[i]))

        for j in range(int(len(list_dir_im))):
            path_to_im = path_im + '/' + list_dir[i] + '/' + list_dir_im[j]
            list_dir2 = sorted(os.listdir(path_to_im))
            path_to_hr = path_hr + '/' + list_dir[i] + '/' + list_dir_hr[j]
            list_dir_hr2 = os.listdir(path_to_hr)
            pulse_rate_file = [filename for filename in list_dir_hr2 if filename.startswith("Pulse")]
            batches_hr = []
            Heart_Rate = []
            batch_overlap = []
            im_path = []
            for pr in pulse_rate_file:
                pr1 = os.path.join(path_hr + '/' + list_dir[i] + '/' + list_dir_hr[j] + '/' + pr)
                with open(pr1, 'r') as file:
                    hr = [line.rstrip('\n') for line in file]
                    batches_hr.append(hr)
            heart_rate = [np.array(pr2).astype(np.float32) for pr2 in batches_hr]
            #print(len(heart_rate[0]), len(list_dir2))
            for im in list_dir2:
                im_dir = path_im + '/' + list_dir[i] + '/' + list_dir_im[j] + '/' + im
                im_path.append(im_dir)

            #print(im_path)
            for l in range(len(batches_hr)):
                B = batches_hr[l]
                C = len(im_path)
                xx = len(B) - C
                #print(xx, C ,len(B))
                if xx > 0 :
                  B= B[0:C]
                elif xx < 0 :
                  for test in range(-xx) :
                    im_path[l].pop()

                xx = len(B) - len(im_path)
                #print("after =" ,xx, len(list_dir2),len(B))

            ########################## overlapping ##############################
            overlapping = 10
            y = B
            for k in range((len(y) - frames_per_step) // overlapping):
                batches_hr = y[k * overlapping: k * overlapping + frames_per_step]
                for b in batches_hr:
                    Heart_Rate.append(b)
            #for m in im_path:

            for n in range((len(im_path) - frames_per_step + overlapping) // overlapping):
                batch = im_path[n * overlapping: n * overlapping + frames_per_step]
                for x in batch:
                    batch_overlap.append(x)
                print(batch)
path_im = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/X-iPPGNet experiments/MMSE-cross/ROI'
path_hr = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/X-iPPGNet experiments/MMSE-cross/HR'

prediction(path_im, path_hr)

