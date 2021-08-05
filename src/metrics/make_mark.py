# from sklearn.metrics import precision_recall_curve,f1_score,auc,roc_curve
# import matplotlib.pyplot as plt
# import numpy as np
#
# precision, recall, thresholds = precision_recall_curve(test_mask.flatten(),pred.flatten())
# # f1 = f1_score(test_mask.flatten(),pred.flatten())
#
# fscore = (2 * precision * recall) / (precision + recall)
# # locate the index of the largest f score
# ix = np.argmax(fscore)
# thr_prerec = thresholds[ix]
# print('Best Threshold=%f, F-Score=%f' % (thresholds[ix], fscore[ix]))
#
# aucpr = auc(recall, precision)
# print('AUCPR=%f'%aucpr)
#
# plt.plot(recall,precision,label='trUnet')
# plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
# plt.grid(True)
# plt.xlabel('recall')
# plt.ylabel('precision')
# no_skill = len(test_mask.flatten()[test_mask.flatten()==1]) / len(test_mask.flatten())
# plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
# plt.legend()
#
# plt.figure()
# fpr, tpr, thresholds = roc_curve(test_mask.flatten(),pred.flatten())
#
# gmeans = np.sqrt(tpr * (1-fpr))
# # locate the index of the largest g-mean
# ix = np.argmax(gmeans)
# print('Best Threshold=%f, G-Mean=%f' % (thresholds[ix], gmeans[ix]))
#
# aucroc = auc(fpr, tpr)
# print('AUCROC=%f'%aucroc)
#
# # J = tpr - fpr
# # ix = np.argmax(J)
# # best_thresh = thresholds[ix]
# # print('Best Threshold=%f, J=%f' % (best_thresh,J[ix]))
#
# plt.plot(fpr, tpr, marker='', label='trUnet')
# plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
# plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.grid(True)
#
#
# def dice_metr_np(thresh):
#     smooth = 1e-3
#
#     def dice_coef_np(y_true, y_pred):
#         #         y_pred = (np.sign(y_pred - thresh) + 1) / 2
#         y_true_f = y_true.flatten()
#         y_pred_f = y_pred.flatten()
#         intersection = np.sum(y_true_f * y_pred_f)
#         return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
#
#     return dice_coef_np
#
#
# dice_coef_np = dice_metr_np(thresh=1)
#
# N = pred.shape[0]
# D = np.arange(0, 1.02, 0.02)
# n = D.shape[0]
# DSC0 = np.zeros((N, n))
# cartilage0 = np.zeros((N, n))
# for k in range(0, n):
#     tr_pred = pred > D[k]
#     for i in range(0, N):
#         DSC0[i, k] = dice_coef_np(tr_pred[i, :, :, 0], test_mask[i, :, :, 0])
#         cartilage0[i, k] = np.sum(test_mask[i, :, :, 0])
#
# print('Mean DSC: ', np.max(np.mean(DSC0, axis=0)))
# # 97 mm * 120 mm
# coef1 = 320 / 120
# coef2 = 260 / 97
# v_coef = coef1 * coef2
#
# i = np.argmax(np.mean(DSC0, axis=0))
#
# DSC = np.zeros((N,))
# cartilage = np.zeros((N,))
# d = D[i]
# DSC = DSC0[:, i]
# cartilage = cartilage0[:, i]
# tr_pred = np.array([pred > d], dtype=int)
# tr_pred = numpy.reshape(tr_pred, (tr_pred.shape[1:]))
#
# print('3D DSC: ', dice_coef_np(tr_pred[:, :, :, 0], test_mask[:, :, :, 0]))
# print('STD DSC: ', np.std(DSC))
# print('Treshold: ', d)
#
# from datetime import datetime
#
# file = fileName[0:fileName.find('model')] + 'mean_DSC_log.txt'
# fp = open(file, 'a')
# fp.write("\nTime: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
# fp.write("\nOpened: %s\nmean DSC: %.5f\n3D DSC: %.5f" % (
# fileName, np.mean(DSC), dice_coef_np(tr_pred[:, :, :, 0], test_mask[:, :, :, 0])))
# fp.write("\nDataset folder: %s\n" % (dataFold))
# fp.write("\nVolume of groud-truth: %g\nVolume of predicted: %g" % (
# np.sum(test_mask[:, :, :, 0]), np.sum(tr_pred[:, :, :, 0])))
# fp.close()
#
# from matplotlib.lines import Line2D
#
# # d=0.60
# d = thr_prerec
# n = 260
# tr_pred = np.array([pred > d], dtype=int)
# tr_pred = numpy.reshape(tr_pred, (tr_pred.shape[1:]))
#
# test_imag0 = np.copy(test_imag)
# # test_imag0 = (test_imag0-np.min(test_imag0))/(np.max(test_imag0)-np.min(test_imag0))
#
# img_rgb = np.copy(test_imag)
# img_rgb = np.append(img_rgb, np.copy(test_imag), axis=3)
# img_rgb = np.append(img_rgb, np.copy(test_imag), axis=3)
# print(img_rgb.shape, np.max(img_rgb), np.min(img_rgb))
#
# f1 = plt.figure(figsize=(12, 50))
# a1 = plt.subplot(121)
# a1.set_title('Ground truth')
#
# img_rgb2 = np.copy(img_rgb)
# # img_rgb2[test_mask[...,0] == 1,:] *=0.7
# img_rgb2[..., 1] = img_rgb[..., 1] + test_mask[..., 0]
# # img_rgb2[...,1] = img_rgb[...,1] + 0.3*test_mask[...,0]
#
# img_rgb2[img_rgb2 > 1] = 1
# plt.imshow(img_rgb2[n, :, :, :])
#
# a2 = plt.subplot(122)
# a2.set_title('Prediction (DSC = {%.5f})' % dice_metr_np(thresh=d)(tr_pred[n, :, :, 0], test_mask[n, :, :, 0]))
# img_rgb2[..., 1] = img_rgb[..., 1] + tr_pred[..., 0]
#
# img_rgb2[img_rgb2 > 1] = 1
#
# plt.imshow(img_rgb2[n, :, :, :])
#
# TP = tr_pred * test_mask  # test_mask == 1 and tr_pred == 1
# FP = ((test_mask * 2 - 2) / (-2)) * tr_pred  # test_mask == 0 and tr_pred == 1
# FN = ((tr_pred * 2 - 2) / (-2)) * test_mask  # test_mask == 1 0 and tr_pred == 0
# img_rgb3 = 1 * np.copy(img_rgb)
# img_rgb3[..., 1] += np.abs(TP[..., 0])  # green
# img_rgb3[..., 0] += np.abs(FP[..., 0])  # red
# img_rgb3[..., 2] += np.abs(FN[..., 0])  # blue
# # img_rgb3 = (img_rgb3-np.min(img_rgb3))/(np.max(img_rgb3)-np.min(img_rgb3))
# img_rgb3[img_rgb3 > 1] = 1
# # img_rgb3[img_rgb3<0] = 0
#
# f2 = plt.figure(figsize=(10, 10))
# legend_elements = [Line2D([0], [0], marker='s', color='w', label='FP',
#                           markerfacecolor='r', markersize=15),
#                    Line2D([0], [0], marker='s', color='w', label='TP',
#                           markerfacecolor='g', markersize=15),
#                    Line2D([0], [0], marker='s', color='w', label='FN',
#                           markerfacecolor='b', markersize=15)]
#
# plt.legend(handles=legend_elements, loc='best')
# plt.imshow(img_rgb3[n, :, :, :])
# # plt.imshow(FP[n,:,:,0])
#
#
# # N = pred.shape[0]
# # DSC = np.zeros((N,))
# # cartilage = np.zeros((N,))
#
#
# # for i in range(0,N):
# #     DSC[i] = dice_metr_np(thresh=d)(tr_pred[i,:,:,0],test_mask[i,:,:,0])
# #     cartilage[i] = np.sum(tr_pred[i,:,:,0])
#
#
# print('DSC on the full test set: ', np.mean(DSC), '\n')
# args = np.argwhere(np.invert(DSC < 1 - 1e-4))
# print('DSC on the full test set without 0s and 1s: ', np.mean(np.delete(DSC, args)), '\n')
#
# # print(np.argwhere(np.abs(DSC < 1e-4)),'\n\n',np.argwhere(np.abs(DSC > 1-1e-4)))
# import pandas as pd
# import re
#
# DSC_pd0 = pd.DataFrame([])
# DSC_3D = np.empty((0,))
# prev = 0
# k = 0
# for i in range(pat_len.shape[0]):
#     if pat_len[i] != 0:
#         print('Case: ', i + 1)
#         print(np.arange(1, pat_len[i] + 1))
#         print(np.arange(prev, prev + pat_len[i]), '\n')
#
#         DSC_pd = pd.DataFrame(np.array(DSC[prev:prev + pat_len[i]]), columns=['DSC,Case {0}'.format(i + 1)])
#         DSC_pd0 = pd.concat([DSC_pd0, DSC_pd], axis=1)
#         print(DSC[prev:prev + pat_len[i]])
#         f3 = plt.figure(figsize=(10, 5))
#         ax1, = plt.plot(range(1, pat_len[i] + 1), DSC[prev:prev + pat_len[i]], 'bo-', label='DSC')
#         plt.grid(True);
#         DSC_3D = np.append(DSC_3D, dice_coef_np(tr_pred[prev:prev + pat_len[i], :, :, 0],
#                                                 test_mask[prev:prev + pat_len[i], :, :, 0]))
#         #         print(DSC_3D,k)
#         plt.title('Case {0}, 3D DSC: {1:.4f}'.format(i + 1, DSC_3D[k]))
#         k += 1
#         #         plt.title('Case {0} DSC: {1:.4f}'.format(i,np.mean(DSC[prev:prev+pat_len[i]])))
#         plt.xlim([0, pat_len[i] + 1 + 1]);
#         plt.ylim([0, 1.2]);
#         plt.xlabel('№ of slice');
#         plt.ylabel('DSC, Cartilage')
#         maxx = max(cartilage[prev:prev + pat_len[i]])
#         #         ax2, = plt.plot(range(1,pat_len[i]+1),cartilage[prev:prev+pat_len[i]]/maxx,'rd--',label='Cartilage tissue')
#
#         prev += pat_len[i]
#         #         plt.legend(handles=(ax1))
#         plt.savefig('Patient {0}_bone.svg'.format(i + 1), format='svg')
#
# s = re.findall('/(\w+\W\w+)/', fileName)
# # DSC_pd0.to_excel('DSC_unet_al_{0}_{1}.xlsx'.format(s,'dataFold'))
# DSC_pd0.to_excel('DSC_unet_al.xlsx')
#
# print('Mean 3D DSC: ', np.mean(DSC_3D))
# print('STD 3D DSC: ', np.std(DSC_3D))from matplotlib.lines import Line2D
#
# # d=0.60
# d=thr_prerec
# n = 260
# tr_pred = np.array([pred > d], dtype=int)
# tr_pred = numpy.reshape(tr_pred,(tr_pred.shape[1:]))
#
# test_imag0 = np.copy(test_imag)
# # test_imag0 = (test_imag0-np.min(test_imag0))/(np.max(test_imag0)-np.min(test_imag0))
#
# img_rgb = np.copy(test_imag)
# img_rgb = np.append(img_rgb,np.copy(test_imag),axis=3)
# img_rgb = np.append(img_rgb,np.copy(test_imag),axis=3)
# print(img_rgb.shape,np.max(img_rgb),np.min(img_rgb))
#
# f1 = plt.figure(figsize=(12,50))
# a1 = plt.subplot(121)
# a1.set_title('Ground truth')
#
# img_rgb2 = np.copy(img_rgb)
# # img_rgb2[test_mask[...,0] == 1,:] *=0.7
# img_rgb2[...,1] = img_rgb[...,1] + test_mask[...,0]
# # img_rgb2[...,1] = img_rgb[...,1] + 0.3*test_mask[...,0]
#
# img_rgb2[img_rgb2>1] = 1
# plt.imshow(img_rgb2[n,:,:,:])
#
# a2 = plt.subplot(122)
# a2.set_title('Prediction (DSC = {%.5f})' % dice_metr_np(thresh=d)(tr_pred[n,:,:,0],test_mask[n,:,:,0]))
# img_rgb2[...,1] = img_rgb[...,1] + tr_pred[...,0]
#
# img_rgb2[img_rgb2>1] = 1
#
# plt.imshow(img_rgb2[n,:,:,:])
#
# TP = tr_pred*test_mask # test_mask == 1 and tr_pred == 1
# FP = ((test_mask*2-2)/(-2)) * tr_pred # test_mask == 0 and tr_pred == 1
# FN = ((tr_pred*2-2)/(-2)) * test_mask# test_mask == 1 0 and tr_pred == 0
# img_rgb3 = 1*np.copy(img_rgb)
# img_rgb3[...,1] += np.abs(TP[...,0]) #green
# img_rgb3[...,0] += np.abs(FP[...,0]) #red
# img_rgb3[...,2] += np.abs(FN[...,0]) #blue
# # img_rgb3 = (img_rgb3-np.min(img_rgb3))/(np.max(img_rgb3)-np.min(img_rgb3))
# img_rgb3[img_rgb3>1] = 1
# # img_rgb3[img_rgb3<0] = 0
#
# f2 = plt.figure(figsize=(10,10))
# legend_elements = [ Line2D([0], [0], marker='s', color='w', label='FP',
#                           markerfacecolor='r', markersize=15),
#                    Line2D([0], [0], marker='s', color='w', label='TP',
#                           markerfacecolor='g', markersize=15),
#                     Line2D([0], [0], marker='s', color='w', label='FN',
#                           markerfacecolor='b', markersize=15)]
#
# plt.legend(handles=legend_elements, loc='best')
# plt.imshow(img_rgb3[n,:,:,:])
# # plt.imshow(FP[n,:,:,0])
#
#
#
#
# # N = pred.shape[0]
# # DSC = np.zeros((N,))
# # cartilage = np.zeros((N,))
#
#
# # for i in range(0,N):
# #     DSC[i] = dice_metr_np(thresh=d)(tr_pred[i,:,:,0],test_mask[i,:,:,0])
# #     cartilage[i] = np.sum(tr_pred[i,:,:,0])
#
#
# print('DSC on the full test set: ',np.mean(DSC),'\n')
# args = np.argwhere( np.invert(DSC < 1-1e-4) )
# print('DSC on the full test set without 0s and 1s: ',np.mean(np.delete(DSC, args )),'\n')
#
# # print(np.argwhere(np.abs(DSC < 1e-4)),'\n\n',np.argwhere(np.abs(DSC > 1-1e-4)))
# import pandas as pd
# import re
#
#
# DSC_pd0 = pd.DataFrame([])
# DSC_3D = np.empty((0,))
# prev = 0
# k = 0
# for i in range(pat_len.shape[0]):
#     if pat_len[i] != 0:
#         print('Case: ',i+1)
#         print(np.arange(1,pat_len[i]+1))
#         print(np.arange(prev,prev+pat_len[i]),'\n')
#
#         DSC_pd = pd.DataFrame(np.array(DSC[prev:prev+pat_len[i]]),columns=['DSC,Case {0}'.format(i+1)])
#         DSC_pd0 = pd.concat([DSC_pd0,DSC_pd],axis=1)
#         print(DSC[prev:prev+pat_len[i]])
#         f3 = plt.figure(figsize=(10,5))
#         ax1, = plt.plot(range(1,pat_len[i]+1),DSC[prev:prev+pat_len[i]],'bo-',label='DSC')
#         plt.grid(True);
#         DSC_3D = np.append(DSC_3D,dice_coef_np(tr_pred[prev:prev+pat_len[i],:,:,0],test_mask[prev:prev+pat_len[i],:,:,0]))
# #         print(DSC_3D,k)
#         plt.title('Case {0}, 3D DSC: {1:.4f}'.format(i+1, DSC_3D[k] ))
#         k += 1
# #         plt.title('Case {0} DSC: {1:.4f}'.format(i,np.mean(DSC[prev:prev+pat_len[i]])))
#         plt.xlim([0,pat_len[i]+1 + 1]);plt.ylim([0,1.2]);
#         plt.xlabel('№ of slice'); plt.ylabel('DSC, Cartilage')
#         maxx = max(cartilage[prev:prev+pat_len[i]])
# #         ax2, = plt.plot(range(1,pat_len[i]+1),cartilage[prev:prev+pat_len[i]]/maxx,'rd--',label='Cartilage tissue')
#
#         prev += pat_len[i]
# #         plt.legend(handles=(ax1))
#         plt.savefig('Patient {0}_bone.svg'.format(i+1),format='svg')
#
# s = re.findall('/(\w+\W\w+)/',fileName)
# # DSC_pd0.to_excel('DSC_unet_al_{0}_{1}.xlsx'.format(s,'dataFold'))
# DSC_pd0.to_excel('DSC_unet_al.xlsx')
#
# print('Mean 3D DSC: ', np.mean(DSC_3D))
# print('STD 3D DSC: ', np.std(DSC_3D))
#
# n = 135
#
# f1 = plt.figure(figsize=(12,50))
# a1 = plt.subplot(121)
# a1.set_title('Ground truth')
#
# img_rgb2 = np.copy(img_rgb)
# img_rgb2[...,1] = img_rgb[...,1] + test_mask[...,0]
# img_rgb2[img_rgb2>1] = 1
# plt.imshow(img_rgb2[n,:,:,:])
#
# a2 = plt.subplot(122)
# a2.set_title('Prediction (DSC = {%.5f})' % dice_metr_np(thresh=d)(tr_pred[n,:,:,0],test_mask[n,:,:,0]))
# img_rgb2[...,1] = img_rgb[...,1] + tr_pred[...,0]
# img_rgb2[img_rgb2>1] = 1
# plt.imshow(img_rgb2[n,:,:,:])
#
#
# f2 = plt.figure(figsize=(10,10))
# legend_elements = [ Line2D([0], [0], marker='s', color='w', label='FP',
#                           markerfacecolor='r', markersize=15),
#                    Line2D([0], [0], marker='s', color='w', label='TP',
#                           markerfacecolor='g', markersize=15),
#                     Line2D([0], [0], marker='s', color='w', label='FN',
#                           markerfacecolor='b', markersize=15)]
#
# plt.legend(handles=legend_elements, loc='best')
# plt.imshow(img_rgb3[n,:,:,:])
#
# from scipy.io import savemat
# import os
# from PIL import Image
#
#
# size = 1024, 1024
# THIS_FOLDER = os.getcwd()
#
#
#
# print(img_rgb2.shape,THIS_FOLDER)
# # mdic = {"img_rgb": img_rgb2}
# # savemat(dataFold+'/img_rgb/predWithMask.mat',mdic)
# fold = '\\Test_data_unet'
# # os.mkdir( dataFold + fold)
# for i in range(img_rgb2.shape[0]):
# #     img_rgb3[i,:,:,:] = (img_rgb3[i,:,:,:]-np.min(img_rgb3[i,:,:,:]))/(np.max(img_rgb3[i,:,:,:])-np.min(img_rgb3[i,:,:,:]))
#     plt.imsave(dataFold + fold + '\\TPFP{0}_{1}.jpg'.format(i,DSC[i]),img_rgb3[i,:,:,:])
#     im = Image.open( dataFold + fold + '\\TPFP{0}_{1}.jpg'.format(i,DSC[i]))
#     im_resized = im.resize(size, Image.LANCZOS)
#     im_resized.save( dataFold + fold + '\\TPFP_highres{0}_{1}.jpg'.format(i,DSC[i]))
#
# from scipy.io import savemat
# import os
# from PIL import Image
#
#
# size = 1024, 1024
# THIS_FOLDER = os.getcwd()
# print(pred.shape)
#
# pred_rgb = np.copy(tr_pred)
# d=thr_prerec
# n = 260
# # pred_rgb = np.array([pred_rgb > d], dtype=np.uint8)
# # pred_rgb = numpy.reshape(pred_rgb,(pred_rgb.shape[1:]))
#
# # pred_rgb = np.append(pred_rgb,np.copy(tr_pred),axis=3)
# # pred_rgb = np.append(pred_rgb,np.copy(tr_pred),axis=3)
# # pred_rgb = pred_rgb.astype(dtype=np.uint8)
#
# plt.imshow(pred_rgb[n,...,0],'gray')
#
#
# mdic = {"pred_rgb": pred_rgb}
# mdic2 = {"set_imag_add": set_imag_add}
#
# fold = '\\Test_data_unet_masks_th'
# # os.mkdir( dataFold + fold)
#
# savemat(dataFold + fold,mdic)
# savemat(dataFold + fold +'_before.mat',mdic2)
#
# for i in range(pred.shape[0]):
# #     img_rgb3[i,:,:,:] = (img_rgb3[i,:,:,:]-np.min(img_rgb3[i,:,:,:]))/(np.max(img_rgb3[i,:,:,:])-np.min(img_rgb3[i,:,:,:]))
#     plt.imsave(dataFold + fold + '\\TPFP{0}.jpg'.format(i),pred_rgb[i,...,0],cmap='gray')
#     im = Image.open( dataFold + fold + '\\TPFP{0}.jpg'.format(i))
# #     im_resized = im.resize(size, Image.LANCZOS)
# #     im_resized.save( dataFold + fold + '\\TPFP_highres{0}.jpg'.format(i))
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# trainDSC = np.load('good2dNet/trUnet2_3/hist_train_dice_coef.npy')
# validDSC = np.load('good2dNet/trUnet2_3/hist_valid_dice_coef.npy')
# trainLoss = np.load('good2dNet/trUnet2_3/hist_train_loss.npy')
# validLoss = np.load('good2dNet/trUnet2_3/hist_valid_loss.npy')
#
# x0 = 0
# x1 = 250
# plt.figure(figsize=(8,7))
# plt.grid(True)
# plt.plot(trainDSC,label='Train')
# plt.plot(validDSC,label='Valid')
# plt.xlabel('№ Epoch'); plt.ylabel('DSC')
# # plt.xlim([x0, x1])
# plt.legend()
# plt.savefig('DSC.svg',format='svg')
# plt.figure(figsize=(8,7))
# # plt.xlim([x0, x1])
# plt.grid(True)
# plt.plot(trainLoss,label='Train')
# plt.plot(validLoss,label='Valid')
# # plt.ylim([0, 0.3])
# plt.xlabel('№ Epoch'); plt.ylabel('Loss')
# plt.legend()
# plt.savefig('Loss.svg',format='svg')
#
#
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# def parseTxt(file):
#     f = open(file, "r")
#     key1 = 'val_dice_coef'
#     key2 = 'val_loss'
#     values = ['loss','dice_coef','val_loss','val_dice_coef']
#     ret = {}
#     i = 0
#     for line in f:
#         if key1 in line and key2 in line:
#             for value in values:
#                 result = re.findall('\\b'+value+'\\b:\s(\d+\.\S+)', line)
#                 ret[value,i] = float(result[0])
# #                 print(float(result[0]))
# #                 print(line)
#             i += 1
#     f.close()
#     return ret,i
#
# file = 'D:/new_SRW/net/good2dNet/trUnet_dice_eucl0/Новый текстовый документ.txt'
# stat,epoch = parseTxt(file)
#
# trainDSC = np.array([stat['dice_coef',i] for i in range(epoch)])
# validDSC = np.array([stat['val_dice_coef',i] for i in range(epoch)])
# trainLoss = np.array([stat['loss',i] for i in range(epoch)])
# validLoss = np.array([stat['val_loss',i] for i in range(epoch)])
#
# a = {}
# a.update({'a':[1,2,3],'b':[1,2,3]})
# print(a)
# a['asd',5]=2
# a['asd',6]=2
# a
#
#
#
# from scipy.io import savemat
# import os
# from PIL import Image
#
#
# size = 1024, 1024
# THIS_FOLDER = os.getcwd()
#
# print(img_rgb2.shape,THIS_FOLDER)
# # mdic = {"img_rgb": img_rgb2}
# # savemat(dataFold+'/img_rgb/predWithMask.mat',mdic)
# fold = '\\Test2_data_unet'
# os.mkdir(THIS_FOLDER + '\\' + dataFold + fold)
# for i in range(img_rgb2.shape[0]):
# #     legend_elements = [ Line2D([0], [0], marker='s', color='w', label='FP',
# #                           markerfacecolor='r', markersize=15),
# #                    Line2D([0], [0], marker='s', color='w', label='TP',
# #                           markerfacecolor='g', markersize=15),
# #                     Line2D([0], [0], marker='s', color='w', label='FN',
# #                           markerfacecolor='b', markersize=15)]
#
# #     plt.legend(handles=legend_elements, loc='best')
# # plt.imshow(img_rgb3[n,:,:,:])
#
# #     plt.imsave(THIS_FOLDER + '\\' + dataFold + fold + '\\{0}_imag.jpg'.format(i),img_rgb[i,:,:,:])
# #     plt.imsave(THIS_FOLDER + '\\' + dataFold + fold + '\\{0}_imag+mask.jpg'.format(i),img_rgb2[i,:,:,:])
# #     plt.imsave(THIS_FOLDER + '\\' + dataFold + fold + '\\{0}_mask.jpg'.format(i),test_mask[i,:,:,0])
#     plt.imsave(THIS_FOLDER + '\\' + dataFold + fold + '\\{0}_TPFP_DSC_{1:f}.jpg'.format(i,DSC[i]),img_rgb3[i,:,:,:])
# #     im = Image.open(THIS_FOLDER + '\\' + dataFold + fold + '\\{0}.jpg'.format(i))
# #     im_resized = im.resize(size, Image.LANCZOS)
# #     im_resized.save(THIS_FOLDER + '\\' + dataFold + fold + '\\TPFP_highres{0}.jpg'.format(i))
#
#
# from scipy.io import savemat
# import os
# from PIL import Image
#
#
# size = 1024, 1024
# THIS_FOLDER = os.getcwd()
#
# print(img_rgb2.shape,THIS_FOLDER)
# # mdic = {"img_rgb": img_rgb2}
# # savemat(dataFold+'/img_rgb/predWithMask.mat',mdic)
# fold = 'Test2_data_unet'
# os.mkdir(dataFold + fold)
# for i in range(img_rgb2.shape[0]):
#     plt.imsave( dataFold + fold + '\\TPFP{0}.jpg'.format(i),img_rgb3[i,:,:,:])
#     im = Image.open( dataFold + fold + '\\TPFP{0}.jpg'.format(i))
#     im_resized = im.resize(size, Image.LANCZOS)
#     im_resized.save( dataFold + fold + '\\TPFP_highres{0}.jpg'.format(i))
#
#
#
# import albumentations as albu
# import cv2
#
# def aug_transforms():
#     return [
#         albu.VerticalFlip(p=0.7),
#         albu.HorizontalFlip(p=0.7),
#         albu.Rotate (limit=180, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, value=None, mask_value=None, always_apply=False, p=0.6),
#         albu.ElasticTransform (alpha=10, sigma=50, alpha_affine=28,
#                                interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, value=None,
#                                mask_value=None, always_apply=False, approximate=False, p=0.6),
#         albu.GridDistortion (num_steps=20, distort_limit=0.2, interpolation=cv2.INTER_LANCZOS4,
#                              border_mode=cv2.BORDER_WRAP, value=None, mask_value=None,
#                              always_apply=False, p=0.5)
#     ]
#
#
#
# transforms = albu.Compose(aug_transforms())
#
#
# n = 50
#
# a = transforms(image=test_imag[n,:,:,0],mask=test_mask[n,:,:,0])
# print(a['image'].shape)
# im = a['image']
# im = np.reshape(im,(im.shape[0],im.shape[1],1))
# mask = a ['mask']
# img_rgb = np.copy(im)
# img_rgb = np.append(img_rgb,im,axis=2)
# img_rgb = np.append(img_rgb,im,axis=2)
# img_rgb[...,1] = img_rgb[...,1] + mask
# img_rgb[img_rgb>1] = 1
# img_rgb[img_rgb<0] = 0
#
# img_rgb2 = test_imag
# img_rgb2 = np.append(img_rgb2,test_imag,axis=3)
# img_rgb2 = np.append(img_rgb2,test_imag,axis=3)
# img_rgb2[...,1] = img_rgb2[...,1] + test_mask[...,0]
# img_rgb2[img_rgb2>1] = 1
#
# plt.imshow(img_rgb)
# plt.figure()
# plt.imshow(img_rgb2[n,:,:,:])
#
# plt.imsave('augmented.png',img_rgb)
# plt.imsave('original.png',img_rgb2[n,:,:,:])
