from PIL import Image
import numpy as np
import math
import pathlib
import skimage.feature as ft
from scipy.spatial.distance import euclidean
import time

def DifBoxCounting(img, r, count):

    (width, height) = img.size
    aparitie = np.array([])
    M = width
    pixel = img.load()
    IMAX = 256
    Nr = 0
    nrPatrate = math.ceil(M / r)
    s = IMAX * (r / M)
    grila = np.zeros((nrPatrate, nrPatrate), dtype='int32')

    for u in range(nrPatrate):
        for v in range(nrPatrate):
            Imax = 0  # p(u,v)
            Imin = 255  # q(u,v)
            for i in range(u * r, min((u + 1) * r, M)):
                for j in range(v * r, min((v + 1) * r, M)):
                    if pixel[i, j] > Imax:
                        Imax = pixel[i, j]

                    if pixel[i, j] < Imin:
                        Imin = pixel[i, j]

            grila[u, v] = math.ceil(Imax / s) - math.ceil(Imin / s) + 1  # nr(u,v)
            Nr += grila[u, v]
            aparitie = np.append(aparitie, (grila[u, v]))

    unic = np.unique(aparitie)
    (sum1, sum2) = (0, 0)

    for iter in range(len(unic)):
        N = unic[iter]
        nr_cazuri_fav = np.count_nonzero(aparitie == N)
        nr_cazuri_pos = len(aparitie)
        P = nr_cazuri_fav / nr_cazuri_pos
        sum1 = sum1 + N*N*P
        sum2 = sum2 + N*P

    lacun[count] = sum1 / (sum2*sum2)

    return Nr


def analizaFractala(surface, count, image, canal, file_df, file_lacun, vect_clasa1, vect_clasa2, rez_clasif, index_imagine, hist_restrans_canal, lbp_clasa1_canal, lbp_clasa2_canal, tip_procesare, clasif_c1, clasif_c2, I_c1, I_c2, file_contrast):

    # min_I_R_C1 = interval_min_C1[0]
    # min_I_G_C1 = interval_min_C1[1]
    # min_I_B_C1 = interval_min_C1[2]
    # min_lac_B_C1 = interval_min_C1[3]
    # min_C_R_C1 = interval_min_C1[4]
    # min_C_G_C1 = interval_min_C1[5]
    # min_C_B_C1 = interval_min_C1[6]
    #
    # max_I_R_C1 = interval_max_C2[0]
    # max_I_G_C1 = interval_max_C2[1]
    # max_I_B_C1 = interval_max_C2[2]
    # max_lac_B_C1 = interval_max_C2[3]
    # max_C_R_C1 = interval_max_C2[4]
    # max_C_G_C1 = interval_max_C2[5]
    # max_C_B_C1 = interval_max_C2[6]

    while surface > 4:
        Ns = DifBoxCounting(image, int(surface), count)
        count = count + 1
        Nr.append(Ns)
        r.append(surface)
        surface = surface / 2

    lacunList = []

    for key, value in lacun.items():
        lacunList.append(value)

    avglacun = sum(lacunList) / len(lacunList)
    avglacun = round(avglacun, 3)
    print("Lacunaritatea pe canalul", canal, ":", avglacun)
    file_lacun.write("%s\t" % round(avglacun, 3))
    #file.write("%s %s %s %s\n" % ("Lacunaritatea pe canalul", canal, ":", avglacun))

    y = np.log(np.array(Nr))
    x = np.log(1 / np.array(r))
    (D, b) = np.polyfit(x, y, deg=1)

    #print("Dimensiunea fractala diferentiala pe canalul", canal, ":", D)
    #file_df.write("%s\t" % D)
    #file.write("%s %s %s %s\n" % ("Dimensiunea fractala diferentiala pe canalul", canal, ":", D))

    I_med = np.mean(image)
    I_med = round(I_med, 3)
    print("Intensitatea medie pe canalul", canal, ":", I_med)
    file_df.write("%s\t" % round(I_med, 3))

    min_c = np.min(image)
    max_c = np.max(image)
    #print("min: ", min_c)
    #print("max: ", max_c)
    numitor = int(max_c) - int(min_c)
    numarator = int(max_c) + int(min_c)
    contrast = numitor/numarator
    file_contrast.write("%s\t" % round(contrast, 3))
    print("Contrast pe canalul", canal, ":", contrast)

    val_df_c1_R = vect_clasa1[0]
    val_df_c1_G = vect_clasa1[1]
    val_df_c1_B = vect_clasa1[2]
    val_df_c2_R = vect_clasa2[0]
    val_df_c2_G = vect_clasa2[1]
    val_df_c2_B = vect_clasa2[2]

    val_lac_c1_R = vect_clasa1[3]
    val_lac_c1_G = vect_clasa1[4]
    val_lac_c1_B = vect_clasa1[5]
    val_lac_c2_R = vect_clasa2[3]
    val_lac_c2_G = vect_clasa2[4]
    val_lac_c2_B = vect_clasa2[5]

    val_contrast_c1_R = vect_clasa1[6]
    val_contrast_c1_G = vect_clasa1[7]
    val_contrast_c1_B = vect_clasa1[8]
    val_contrast_c2_R = vect_clasa2[6]
    val_contrast_c2_G = vect_clasa2[7]
    val_contrast_c2_B = vect_clasa2[8]

    if canal == 'R':
        dist_df_c1 = abs(I_med - val_df_c1_R)
        dist_df_c2 = abs(I_med - val_df_c2_R)
        dist_lac_c1 = abs(avglacun - val_lac_c1_R)
        dist_lac_c2 = abs(avglacun - val_lac_c2_R)
        dist_contrast_c1 = abs(contrast - val_contrast_c1_R)
        dist_contrast_c2 = abs(contrast - val_contrast_c2_R)

    if canal == 'G':
        dist_df_c1 = abs(I_med - val_df_c1_G)
        dist_df_c2 = abs(I_med - val_df_c2_G)
        dist_lac_c1 = abs(avglacun - val_lac_c1_G)
        dist_lac_c2 = abs(avglacun - val_lac_c2_G)
        dist_contrast_c1 = abs(contrast - val_contrast_c1_G)
        dist_contrast_c2 = abs(contrast - val_contrast_c2_G)

    if canal == 'B':
        dist_df_c1 = abs(I_med - val_df_c1_B)
        dist_df_c2 = abs(I_med - val_df_c2_B)
        dist_lac_c1 = abs(avglacun - val_lac_c1_B)
        dist_lac_c2 = abs(avglacun - val_lac_c2_B)
        dist_contrast_c1 = abs(contrast - val_contrast_c1_B)
        dist_contrast_c2 = abs(contrast - val_contrast_c2_B)


    if tip_procesare == "retina":
        if dist_df_c1 < dist_df_c2:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind retina pentru clasificatorul intensitate medie, pe canalul", canal))
            clasif_c1 += 1
        else:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind exudat pentru clasificatorul intensitate medie, pe canalul", canal))
            clasif_c2 += 1

        if dist_lac_c1 < dist_lac_c2:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind retina pentru clasificatorul lacunaritate, pe canalul", canal))
            clasif_c1 += 1
        else:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind exudat pentru clasificatorul lacunaritate, pe canalul", canal))
            clasif_c2 += 1

        if dist_contrast_c1 < dist_contrast_c2:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
                                                "a fost clasificat ca fiind retina pentru clasificatorul contrast pe canalul", canal))
            clasif_c1 += 1
        else:
            clasif_c2 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
                                                "a fost clasificat ca fiind exudat pentru clasificatorul contrast pe canalul", canal))

    if tip_procesare == "paduri":
        if dist_df_c1 < dist_df_c2:
            clasif_c1 += 1
            I_c1 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind padure pentru clasificatorul intensitate medie, pe canalul", canal))
        else:
            clasif_c2 += 1
            I_c2 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind camp pentru clasificatorul intensitate medie, pe canalul", canal))

        if dist_lac_c1 < dist_lac_c2 and canal == 'B':
            clasif_c1 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind padure pentru clasificatorul lacunaritate, pe canalul", canal))
        if dist_lac_c1 > dist_lac_c2 and canal == 'B':
            clasif_c2 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine, "a fost clasificat ca fiind camp pentru clasificatorul lacunaritate, pe canalul", canal))

        dist_lbp_clasa1_canal = euclidean(hist_restrans_canal, lbp_clasa1_canal)
        dist_lbp_clasa2_canal = euclidean(hist_restrans_canal, lbp_clasa2_canal)
        # if dist_lbp_clasa1_canal < dist_lbp_clasa2_canal:
        #     rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
        #                                      "a fost clasificat ca fiind padure pentru clasificatorul histograma LBP pe canalul", canal))
        #     clasif_c1 += 1
        # else:
        #     clasif_c2 += 1
        #     rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
        #                                      "a fost clasificat ca fiind camp pentru clasificatorul histograma LBP pe canalul", canal))

        if dist_contrast_c1 < dist_contrast_c2:
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
                                                "a fost clasificat ca fiind padure pentru clasificatorul contrast pe canalul", canal))
            clasif_c1 += 1
        else:
            clasif_c2 += 1
            rez_clasif.write("%s %s %s %s\n" % ("Patchul", index_imagine,
                                                "a fost clasificat ca fiind camp pentru clasificatorul contrast pe canalul", canal))


    return D, avglacun, clasif_c1, clasif_c2, I_c1, I_c2


def procesarePatchuri(path, file_df, file_lacun, file_lbp, format, vect_clasa1, vect_clasa2, rez_clasif, lbp_clasa1_R, lbp_clasa2_R, lbp_clasa1_G, lbp_clasa2_G, lbp_clasa1_B, lbp_clasa2_B, tip_procesare, nr_imagine, file_contrast):

    files = pathlib.Path(path)
    index_imagine = 0
    for image in files.glob(format):
        clasif_c1 = 0
        clasif_c2 = 0
        I_c1 = 0
        I_c2 = 0
        all = False
        index_imagine += 1
        imageLBP = image
        img = Image.open(image).convert('RGB')
        image = Image.open(image)
        r1, g1, b1 = img.split()
        r1 = r1.point(lambda i: i * 1.5)
        g1 = g1.point(lambda i: i * 0.5)
        b1 = g1.point(lambda i: i * 0.5)
        image_gray = Image.merge('RGB', (r1,g1,b1))

        try:
            (r, g, b) = image.split()
        except Exception as error:
            print("Eroare: ", repr(error))
        else:
            image1 = r
            LBP = ft.local_binary_pattern(r, 8, 1)
            hist, _ = np.histogram(LBP, bins=np.arange(2 ** 8 + 1), density=True)
            hist_restrans_r = restrangereLBP(hist, 25)
            image2 = g
            LBP = ft.local_binary_pattern(g, 8, 1)
            hist, _ = np.histogram(LBP, bins=np.arange(2 ** 8 + 1), density=True)
            hist_restrans_g = restrangereLBP(hist, 25)
            image3 = b
            LBP = ft.local_binary_pattern(b, 8, 1)
            hist, _ = np.histogram(LBP, bins=np.arange(2 ** 8 + 1), density=True)
            hist_restrans_b = restrangereLBP(hist, 25)
            (width, height) = image.size

            # print("%s: %s" % ("histograma LBP", hist))

            ################################################# IMAG1
            # if index_imagine >= 1 and index_imagine <= 10:
            #     lbparr_R.append(hist_restrans_r)
            #     lbparr_G.append(hist_restrans_g)
            #     lbparr_B.append(hist_restrans_b)
            # if index_imagine == 67 or index_imagine == 68 or index_imagine == 86 or index_imagine == 87 or index_imagine == 88 or index_imagine == 89 or index_imagine == 104 or index_imagine == 105 or index_imagine == 106 or index_imagine == 107:
            #     lbparrcamp_R.append(hist_restrans_r)
            #     lbparrcamp_G.append(hist_restrans_g)
            #     lbparrcamp_B.append(hist_restrans_b)
            ################################################## IMAG2
            # if index_imagine >= 1 and index_imagine <= 10:
            #     lbparr_R.append(hist_restrans_r)
            #     lbparr_G.append(hist_restrans_g)
            #     lbparr_B.append(hist_restrans_b)
            # if index_imagine == 164 or index_imagine == 165 or index_imagine == 166 or index_imagine == 167 or index_imagine == 168 or index_imagine == 204 or index_imagine == 205 or index_imagine == 206 or index_imagine == 207 or index_imagine == 208:
            #     lbparrcamp_R.append(hist_restrans_r)
            #     lbparrcamp_G.append(hist_restrans_g)
            #     lbparrcamp_B.append(hist_restrans_b)
            # ################################################## IMAG3
            # if index_imagine >= 1 and index_imagine <= 10:
            #     lbparr_R.append(hist_restrans_r)
            #     lbparr_G.append(hist_restrans_g)
            #     lbparr_B.append(hist_restrans_b)
            # if index_imagine == 93 or index_imagine == 94 or index_imagine == 95 or index_imagine == 98 or index_imagine == 113 or index_imagine == 114 or index_imagine == 115 or index_imagine == 119 or index_imagine == 120 or index_imagine == 154:
            #     lbparrcamp_R.append(hist_restrans_r)
            #     lbparrcamp_G.append(hist_restrans_g)
            #     lbparrcamp_B.append(hist_restrans_b)
            # ################################################## IMAG4
            # if index_imagine >= 1 and index_imagine <= 10:
            #     lbparr_R.append(hist_restrans_r)
            #     lbparr_G.append(hist_restrans_g)
            #     lbparr_B.append(hist_restrans_b)
            # if index_imagine == 181 or index_imagine == 182 or index_imagine == 183 or index_imagine == 184 or index_imagine == 185 or index_imagine == 186 or index_imagine == 201 or index_imagine == 204 or index_imagine == 205 or index_imagine == 206:
            #     lbparrcamp_R.append(hist_restrans_r)
            #     lbparrcamp_G.append(hist_restrans_g)
            #     lbparrcamp_B.append(hist_restrans_b)
            # ################################################## IMAG5
            # if index_imagine >= 1 and index_imagine <= 10:
            #     lbparr_R.append(hist_restrans_r)
            #     lbparr_G.append(hist_restrans_g)
            #     lbparr_B.append(hist_restrans_b)
            # if index_imagine == 372 or index_imagine == 373 or index_imagine == 392 or index_imagine == 393 or index_imagine == 352 or index_imagine == 332 or index_imagine == 330 or index_imagine == 331 or index_imagine == 329 or index_imagine == 370:
            #     lbparrcamp_R.append(hist_restrans_r)
            #     lbparrcamp_G.append(hist_restrans_g)
            #     lbparrcamp_B.append(hist_restrans_b)
            ##################################################

            arie_start = width / 2

            D_R, lacun_R, clasif_c1, clasif_c2, I_c1, I_c2 = analizaFractala(arie_start, 1, image1, "R", file_df, file_lacun, vect_clasa1, vect_clasa2, rez_clasif, index_imagine, hist_restrans_r, lbp_clasa1_R, lbp_clasa2_R, tip_procesare, clasif_c1, clasif_c2, I_c1, I_c2, file_contrast)
            D_G, lacun_G, clasif_c1, clasif_c2, I_c1, I_c2 = analizaFractala(arie_start, 1, image2, "G", file_df, file_lacun, vect_clasa1, vect_clasa2, rez_clasif, index_imagine, hist_restrans_g, lbp_clasa1_G, lbp_clasa2_G, tip_procesare, clasif_c1, clasif_c2, I_c1, I_c2, file_contrast)
            D_B, lacun_B, clasif_c1, clasif_c2, I_c1, I_c2 = analizaFractala(arie_start, 1, image3, "B", file_df, file_lacun, vect_clasa1, vect_clasa2, rez_clasif, index_imagine, hist_restrans_b, lbp_clasa1_B, lbp_clasa2_B, tip_procesare, clasif_c1, clasif_c2, I_c1, I_c2, file_contrast)

            print("Imaginea: ", imageLBP)
            avg_df = (D_R + D_G + D_B) / 3
            print("Media dimensiunii fractale: ", avg_df)
            avg_lacun = (lacun_R + lacun_G + lacun_B) / 3
            print("Media lacunaritatii: ", avg_lacun)

            print("Clasificatori C1 pentru patch-ul", index_imagine, ":", clasif_c1)
            print("Clasificatori C2 pentru patch-ul", index_imagine, ":", clasif_c2)

            if tip_procesare == "paduri":
                rez_paduri = open("clasificare_paduri.txt", "a")
                file_name = "paduri_procesare/padure" + str(nr_imagine) + "/padure" + str(index_imagine) + ".png"
                # if I_c1 == 3:
                #     print("Patch-ul ", index_imagine, " apartine clasei padure")
                #     print("Toate 3 canalele - padure, pentru intensitatea medie")
                #     image.save(file_name)
                #     rez_paduri.write("%s %s %s\n" % ("Patch-ul", index_imagine, "apartine clasei padure"))
                #     all = True
                # else:
                #     if I_c2 == 3:
                #         print("Patch-ul ", index_imagine, " apartine clasei camp")
                #         print("Toate 3 canalele - camp, pentru intensitatea medie")
                #         image_gray.save(file_name)
                #         all = True
                if clasif_c1 > clasif_c2 and all == False:
                    print("Patch-ul ", index_imagine, " apartine clasei padure")
                    image.save(file_name)
                    rez_paduri.write("%s %s %s\n" % ("Patch-ul", index_imagine, "apartine clasei padure"))
                else:
                    if clasif_c1 < clasif_c2 and all == False:
                        print("Patch-ul ", index_imagine, " apartine clasei camp")
                        image_gray.save(file_name)

                    rez_paduri.write("%s %s %s\n" % ("Patch-ul", index_imagine, "apartine clasei camp"))
            else:
                if tip_procesare == "retina":
                    rez_retina = open("clasificare_retina.txt", "a")
                    file_name = "retina_procesare/retina" + str(nr_imagine) + "/retina" + str(index_imagine) + ".png"
                    if clasif_c1 > clasif_c2:
                        print("Patch-ul ", index_imagine, " apartine clasei retina")
                        image.save(file_name)
                        rez_retina.write("%s %s %s\n" % ("Patch-ul", index_imagine, "apartine clasei retina"))
                    else:
                        print("Patch-ul ", index_imagine, " apartine clasei exudat")
                        image_gray.save(file_name)
                        rez_retina.write("%s %s %s\n" % ("Patch-ul", index_imagine, "apartine clasei exudat"))

            print("\n")
            #file_df.write("%s\n" % avg_df)
            file_df.write("\n")
            file_contrast.write("\n")
            file_lacun.write("%s\n" % round(avg_lacun, 3))
            hist_r = ["%.3f" % member for member in hist_restrans_r]
            hist_g = ["%.3f" % member for member in hist_restrans_g]
            hist_b = ["%.3f" % member for member in hist_restrans_b]
            file_lbp.write("%s\t %s\t %s\n" % (hist_r, hist_g, hist_b))


def procesarePaduri():

    vectpadure = [77.728, 94.879, 88.317, 1.092, 1.087, 1.084]
    vectcamp = [171.487, 171.193, 125.804, 1.069, 1.066, 1.063]
    lbppad_R = [0.186, 0.09, 0.105, 0.018, 0.095, 0.094, 0.016, 0.104, 0.035, 0.258]
    lbpcamp_R = [0.184, 0.086, 0.103, 0.018, 0.091, 0.094, 0.017, 0.11, 0.036, 0.261]
    lbppad_G = [0.184, 0.091, 0.105, 0.017, 0.095, 0.094, 0.016, 0.104, 0.035, 0.259]
    lbpcamp_G = [0.184, 0.085, 0.103, 0.018, 0.092, 0.094, 0.017, 0.11, 0.036, 0.262]
    lbppad_B = [0.188, 0.088, 0.104, 0.018, 0.092, 0.093, 0.017, 0.104, 0.036, 0.26]
    lbpcamp_B = [0.187, 0.085, 0.103, 0.018, 0.09, 0.094, 0.017, 0.11, 0.036, 0.261]

    # CLASA 1 - PADURE
    # CLASA 2 - CAMP
    vect1padure = [83.718, 107.19, 118.19, 1.099, 1.093, 1.093, 0.9243, 0.784, 0.7192]
    vect1camp = [195.03, 199.483, 159.876, 1.068, 1.069, 1.05, 0.5235, 0.4433, 0.5304]
    lbp1pad_R = [0.194, 0.091, 0.104, 0.018, 0.099, 0.094, 0.016, 0.093, 0.035, 0.256]
    lbp1camp_R = [0.199, 0.076, 0.092, 0.018, 0.103, 0.11, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_G = [0.193, 0.092, 0.105, 0.018, 0.101, 0.095, 0.016, 0.092, 0.035, 0.254]
    lbp1camp_G = [0.199, 0.075, 0.091, 0.018, 0.103, 0.111, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_B = [0.196, 0.09, 0.104, 0.018, 0.096, 0.094, 0.016, 0.094, 0.036, 0.257]
    lbp1camp_B = [0.199, 0.075, 0.092, 0.018, 0.103, 0.109, 0.017, 0.099, 0.034, 0.254]
    # interval_min1padure = [73.31, 96.44, 111.1, 1.07, 0.863, 0.717, 0.644]
    # interval_max1padure = [92.07, 120.65, 128.4, 1.11, 1, 0.903, 0.843]
    # interval_min1camp = [167.51,176.5,152.25, 1.04, 0.328, 0.321, 0.428]
    # intervalmax1camp = [206.87,209.1,167.81, 1.06, 0.759, 0.604, 0.7]
    image1_file = open("image1_df.csv", "w")
    image1_file1 = open("image1_lacun.csv", "w")
    image1_file2 = open("image1_lbp.csv", "w")
    image1_file3 = open("rezultate_clasificare_imagine1.csv", "w")
    image1_file4 = open("image1_contrast.csv", "w")
    path_imagine1 = 'paduri\cropped\patchuri_imagine1'
    procesarePatchuri(path_imagine1, image1_file, image1_file1, image1_file2, '*.jpg', vect1padure, vect1camp, image1_file3, lbp1pad_R, lbp1camp_R, lbp1pad_G, lbp1camp_G, lbp1pad_B, lbp1camp_B, "paduri", 1, image1_file4)
    image1_file.close()
    image1_file1.close()
    image1_file2.close()
    image1_file3.close()
    image1_file4.close()

    vect2padure = [83.883, 100.98, 107.359, 1.095, 1.093, 1.092, 0.8863, 0.774, 0.7415]
    vect2camp = [209.017, 203.287, 161.828, 1.088, 1.085, 1.08, 0.4133, 0.4089, 0.5446]
    lbp2pad_R = [0.194, 0.091, 0.104, 0.018, 0.101, 0.096, 0.016, 0.091, 0.035, 0.254]
    lbp2camp_R = [0.174, 0.09, 0.105, 0.017, 0.102, 0.096, 0.016, 0.109, 0.035, 0.257]
    lbp2pad_G = [0.193, 0.091, 0.104, 0.018, 0.102, 0.097, 0.016, 0.091, 0.035, 0.254]
    lbp2camp_G = [0.174, 0.089, 0.106, 0.017, 0.103, 0.096, 0.016, 0.108, 0.035, 0.256]
    lbp2pad_B = [0.194, 0.091, 0.103, 0.018, 0.1, 0.095, 0.016, 0.092, 0.035, 0.255]
    lbp2camp_B = [0.179, 0.09, 0.106, 0.017, 0.1, 0.095, 0.017, 0.107, 0.035, 0.255]
    # interval_min2padure = [75.89, 92.25, 99.05, 1.06, 0.8, 0.71, 0.61]
    # interval_max2padure = [97.12, 116.884, 116.42, 1.13, 0.97, 0.84, 0.8]
    # interval_min2camp = [194.5, 190.5, 152.9, 1.03, 0.25, 0.28, 0.38]
    # intervalmax2camp = [220.32, 212.25, 166.7, 1.18, 0.59, 0.55, 0.665]
    image2_file = open("image2_df.csv", "w")
    image2_file1 = open("image2_lacun.csv", "w")
    image2_file2 = open("image2_lbp.csv", "w")
    image2_file3 = open("rezultate_clasificare_imagine2.csv", "w")
    image2_file4 = open("image2_contrast.csv", "w")
    path_imagine2 = 'paduri\cropped\patchuri_imagine2'
    procesarePatchuri(path_imagine2, image2_file, image2_file1, image2_file2, '*.jpg', vect2padure, vect2camp, image2_file3, lbp2pad_R, lbp2camp_R, lbp2pad_G, lbp2camp_G, lbp2pad_B, lbp2camp_B, "paduri", 2, image2_file4)
    image2_file.close()
    image2_file1.close()
    image2_file2.close()
    image2_file3.close()
    image2_file4.close()

    # vect3padure = [99.874, 104.331, 83.052, 1.071, 1.074, 1.062, 0.8752, 0.8056, 0.8301]
    # vect3camp = [161.425, 146.633, 100.309, 1.042, 1.04, 1.036, 0.4502, 0.4622, 0.6265]
    # lbp3pad_R = [0.186, 0.079, 0.115, 0.02, 0.088, 0.093, 0.019, 0.115, 0.036, 0.249]
    # lbp3camp_R = [0.196, 0.083, 0.11, 0.02, 0.073, 0.084, 0.019, 0.113, 0.039, 0.263]
    # lbp3pad_G = [0.185, 0.078, 0.115, 0.02, 0.088, 0.093, 0.019, 0.115, 0.036, 0.251]
    # lbp3camp_G = [0.196, 0.082, 0.11, 0.02, 0.073, 0.083, 0.019, 0.113, 0.039, 0.265]
    # lbp3pad_B = [0.19, 0.076, 0.114, 0.02, 0.086, 0.091, 0.019, 0.116, 0.036, 0.252]
    # lbp3camp_B = [0.198, 0.081, 0.109, 0.02, 0.073, 0.082, 0.019, 0.113, 0.039, 0.265]
    # image3_file = open("image3_df.csv", "w")
    # image3_file1 = open("image3_lacun.csv", "w")
    # image3_file2 = open("image3_lbp.csv", "w")
    # image3_file3 = open("rezultate_clasificare_imagine3.csv", "w")
    # image3_file4 = open("image3_contrast.csv", "w")
    # path_imagine3 = 'paduri\cropped\patchuri_imagine3'
    # procesarePatchuri(path_imagine3, image3_file, image3_file1, image3_file2, '*.jpg', vect3padure, vect3camp, image3_file3, lbp3pad_R, lbp3camp_R, lbp3pad_G, lbp3camp_G, lbp3pad_B, lbp3camp_B, "paduri", 3, image3_file4)
    # image3_file.close()
    # image3_file1.close()
    # image3_file2.close()
    # image3_file3.close()
    # image3_file4.close()
    #
    # vect4padure = [54.677, 68.363, 54.425, 1.093, 1.084, 1.082, 0.9838, 0.8639, 0.9177]
    # vect4camp = [155.595, 151.883, 101.985, 1.097, 1.093, 1.1, 0.4458, 0.4128, 0.6223]
    # lbp4pad_R = [0.176, 0.105, 0.088, 0.014, 0.091, 0.092, 0.014, 0.101, 0.037, 0.282]
    # lbp4camp_R = [0.17, 0.096, 0.096, 0.016, 0.096, 0.088, 0.015, 0.107, 0.036, 0.279]
    # lbp4pad_G = [0.175, 0.105, 0.088, 0.014, 0.091, 0.091, 0.013, 0.102, 0.037, 0.283]
    # lbp4camp_G = [0.171, 0.095, 0.095, 0.016, 0.096, 0.088, 0.014, 0.106, 0.036, 0.283]
    # lbp4pad_B = [0.181, 0.1, 0.087, 0.016, 0.088, 0.089, 0.015, 0.099, 0.038, 0.286]
    # lbp4camp_B = [0.176, 0.094, 0.096, 0.017, 0.093, 0.089, 0.015, 0.106, 0.037, 0.278]
    # image4_file = open("image4_df.csv", "w")
    # image4_file1 = open("image4_lacun.csv", "w")
    # image4_file2 = open("image4_lbp.csv", "w")
    # image4_file3 = open("rezultate_clasificare_imagine4.csv", "w")
    # image4_file4 = open("image4_contrast.csv", "w")
    # path_imagine4 = 'paduri\cropped\patchuri_imagine4'
    # procesarePatchuri(path_imagine4, image4_file, image4_file1, image4_file2, '*.jpg', vect4padure, vect4camp, image4_file3, lbp4pad_R, lbp4camp_R, lbp4pad_G, lbp4camp_G, lbp4pad_B, lbp4camp_B, "paduri", 4, image4_file4)
    # image4_file.close()
    # image4_file1.close()
    # image4_file2.close()
    # image4_file3.close()
    # image4_file4.close()
    #
    # vect5padure = [66.488, 93.53, 78.558, 1.101, 1.092, 1.092, 0.9801, 0.8446, 0.8498]
    # vect5camp = [136.37, 154.681, 105.024, 1.048, 1.044, 1.047, 0.8425, 0.707, 0.8644]
    # lbp5pad_R = [0.178, 0.084, 0.111, 0.019, 0.093, 0.095, 0.018, 0.117, 0.035, 0.25]
    # lbp5camp_R = [0.182, 0.084, 0.113, 0.018, 0.082, 0.092, 0.017, 0.123, 0.035, 0.252]
    # lbp5pad_G = [0.175, 0.087, 0.111, 0.018, 0.094, 0.096, 0.017, 0.117, 0.034, 0.251]
    # lbp5camp_G = [0.179, 0.086, 0.112, 0.018, 0.084, 0.094, 0.017, 0.124, 0.035, 0.252]
    # lbp5pad_B = [0.178, 0.082, 0.112, 0.018, 0.093, 0.095, 0.018, 0.118, 0.035, 0.252]
    # lbp5camp_B = [0.183, 0.083, 0.112, 0.019, 0.082, 0.092, 0.017, 0.124, 0.036, 0.252]
    # image5_file = open("image5_df.csv", "w")
    # image5_file1 = open("image5_lacun.csv", "w")
    # image5_file2 = open("image5_lbp.csv", "w")
    # image5_file3 = open("rezultate_clasificare_imagine5.csv", "w")
    # image5_file4 = open("image5_contrast.csv", "w")
    # path_imagine5 = 'paduri\cropped\patchuri_imagine5'
    # procesarePatchuri(path_imagine5, image5_file, image5_file1, image5_file2, '*.jpg', vect5padure, vect5camp, image5_file3, lbp5pad_R, lbp5camp_R, lbp5pad_G, lbp5camp_G, lbp5pad_B, lbp5camp_B, "paduri", 5, image5_file4)
    # image5_file.close()
    # image5_file1.close()
    # image5_file2.close()
    # image5_file3.close()
    # image5_file4.close()

    # arr = [lbp1pad_R, lbp2pad_R, lbp3pad_R, lbp4pad_R, lbp5pad_R]
    # medie(arr)
    # arr = [lbp1camp_R, lbp2camp_R, lbp3camp_R, lbp4camp_R, lbp5camp_R]
    # medie(arr)
    # arr = [lbp1pad_G, lbp2pad_G, lbp3pad_G, lbp4pad_G, lbp5pad_G]
    # medie(arr)
    # arr = [lbp1camp_G, lbp2camp_G, lbp3camp_G, lbp4camp_G, lbp5camp_G]
    # medie(arr)
    # arr = [lbp1pad_B, lbp2pad_B, lbp3pad_B, lbp4pad_B, lbp5pad_B]
    # medie(arr)
    # arr = [lbp1camp_B, lbp2camp_B, lbp3camp_B, lbp4camp_B, lbp5camp_B]
    # medie(arr)
    # arr = [vect1padure, vect2padure, vect3padure, vect4padure, vect5padure]
    # medie(arr)
    # arr = [vect1camp, vect2camp, vect3camp, vect4camp, vect5camp]
    # medie(arr)



def procesareRetina():

    # CLASA 1 = RETINA
    # CLASA 2 = EXUDAT
    # image1_file = open("image1ret_df.txt", "a")
    # image1_file1 = open("image1ret_lacun.txt", "a")
    # image1_file2 = open("image1ret_lbp.csv", "a")
    # image1_file3 = open("clasificare_retina1.csv", "a")
    # path_imagine1 = 'retina\split retina 1'
    # vect1retina = [1.651982202, 1.650432263, 1.648489028, 1.14808068, 1.134484046, 1.079266677]
    # vect1exudat = [1.66597312, 1.666008278, 1.665451825, 1.212790758, 1.123817246, 1.089496454]
    # lbp1retina = [0.032625638269282456, 0.018785272776135446, 0.000913732867508734, 0.005778016662187584, 0.01639344262295082, 0.0028755710830421924, 0.006342381080354742, 0.025611394786347757, 0.0011556033324375168, 0.00040311744154797097, 0.00010749798441279225, 0.00010749798441279225, 0.0062080085998387526, 0.0009406073636119323, 0.008169846815372212, 0.016097823165815643, 0.01918839021768342, 0.007014243482934694, 0.0004837409298575651, 0.0018543402311206666, 0.002445579145391024, 0.00037624294544477287, 0.0009137328675087343, 0.0025530771298038165, 0.0067186240257995165, 0.0017468422467078742, 0.0001343724805159903, 0.0006449879064767537, 0.027492609513571624, 0.002821822090835797, 0.019080892233270627, 0.03155065842515453, 0.0012899758129535073, 0.0005374899220639613, 8.06234883095942e-05, 0.0001612469766191884, 0.0004568664337543671, 0.00010749798441279225, 0.00018812147272238643, 0.0005374899220639614, 8.06234883095942e-05, 0.0, 0.0, 0.0, 0.00018812147272238643, 0.0, 0.0001612469766191884, 0.00040311744154797097, 0.006396130072561139, 0.0014243482934694974, 0.00010749798441279225, 0.00018812147272238643, 0.0007793603869927438, 0.00018812147272238643, 0.0001343724805159903, 0.0006718624025799516, 0.0077129803816178445, 0.0011824778285407147, 0.00018812147272238643, 0.00037624294544477287, 0.018355280838484277, 0.0008062348830959418, 0.01053480247245364, 0.011744154797097554, 0.01690405804891158, 0.0023380811609782317, 0.00029561945713517867, 0.0007793603869927438, 0.006046761623219565, 0.0003224939532383768, 0.0017468422467078742, 0.0024724536414942218, 0.0005374899220639614, 0.0001343724805159903, 2.6874496103198063e-05, 8.06234883095942e-05, 0.0016930932545014782, 8.06234883095942e-05, 0.0014780972856758936, 0.0010212308519215265, 0.0021768341843590435, 0.00037624294544477287, 0.0, 0.00024187046492878255, 0.0002687449610319806, 2.6874496103198063e-05, 8.06234883095942e-05, 2.6874496103198063e-05, 0.0005374899220639614, 5.3748992206396127e-05, 2.6874496103198063e-05, 0.0, 0.0022305831765654393, 0.0001343724805159903, 0.0010749798441279227, 0.0009674818597151304, 0.0069604944907282985, 0.0007524858908895458, 5.3748992206396127e-05, 0.00037624294544477287, 0.0016124697661918839, 5.3748992206396127e-05, 0.0002687449610319807, 0.0005374899220639614, 0.00013437248051599034, 2.6874496103198063e-05, 0.0, 2.6874496103198063e-05, 0.0007793603869927438, 2.6874496103198063e-05, 0.0005912389142703575, 0.0002418704649287826, 0.0263638806772373, 0.002579951625907014, 0.000456866433754367, 0.0008599838753023381, 0.003305563020693362, 8.06234883095942e-05, 0.0007524858908895457, 0.0011556033324375168, 0.021069604944907283, 0.0009674818597151304, 0.00037624294544477287, 0.00045686643375436717, 0.03265251276538565, 0.0011824778285407151, 0.013141628594463856, 0.009110454178984144, 0.0008868583714055361, 0.005643644181671593, 0.00010749798441279225, 0.007578607901101856, 0.0006181134103735554, 0.0007793603869927438, 0.00034936844934157477, 0.019080892233270624, 0.0, 0.00016124697661918836, 2.6874496103198063e-05, 0.0001343724805159903, 5.3748992206396127e-05, 0.00018812147272238643, 0.0002687449610319806, 0.009567320612738512, 0.0004837409298575652, 0.001182477828540715, 0.0, 0.0013705993012631012, 2.6874496103198063e-05, 0.0002149959688255845, 0.0, 0.000913732867508734, 5.3748992206396127e-05, 0.0002149959688255845, 0.0, 0.0006718624025799515, 0.000429991937651169, 0.0007524858908895458, 0.00037624294544477287, 0.010481053480247245, 0.00010749798441279225, 0.0002149959688255845, 2.6874496103198063e-05, 0.0002149959688255845, 0.0, 0.0, 2.6874496103198063e-05, 0.00042999193765116907, 0.0, 2.6874496103198063e-05, 0.0, 2.6874496103198063e-05, 5.3748992206396127e-05, 0.0, 0.0, 0.0003224939532383768, 0.00010749798441279225, 0.0002418704649287826, 2.6874496103198063e-05, 0.0004568664337543671, 0.0, 2.6874496103198063e-05, 2.6874496103198063e-05, 0.00034936844934157477, 0.0003224939532383767, 0.0005374899220639614, 5.3748992206396127e-05, 0.0006987368986831496, 0.000456866433754367, 0.00029561945713517867, 0.0004837409298575652, 0.009406073636119323, 0.006691749529696318, 0.02695511959150766, 0.00024187046492878255, 0.021741467347487234, 0.0015855952700886858, 0.002526202633700618, 0.0009674818597151304, 0.03171190540177372, 8.06234883095942e-05, 0.0005643644181671592, 0.0, 0.0003493684493415748, 0.00021499596882558453, 0.0007256113947863476, 0.0005374899220639614, 0.011824778285407149, 0.0008868583714055361, 0.002096210696049449, 0.0, 0.0010749798441279225, 2.6874496103198063e-05, 0.00018812147272238643, 2.6874496103198063e-05, 0.0007524858908895457, 0.0003493684493415748, 0.0006449879064767537, 0.0, 0.000429991937651169, 0.0008599838753023381, 0.001316850309056705, 0.0003224939532383768, 0.009002956194571353, 0.007954850846546627, 0.019107766729373827, 0.0001612469766191884, 0.010292932007524857, 0.0010481053480247246, 0.0011824778285407147, 0.0003224939532383768, 0.011636656812684762, 0.0002687449610319807, 0.0004568664337543671, 0.0, 0.0004568664337543671, 0.0006718624025799515, 0.00037624294544477287, 0.0009674818597151302, 0.00943294813222252, 0.019779629131953778, 0.02985756517065305, 0.00037624294544477287, 0.012174146734748724, 0.0005106154259607633, 0.0009943563558183284, 0.0003493684493415748, 0.008519215264713786, 0.010507927976350443, 0.011233539371136792, 0.000429991937651169, 0.00927170115560333, 0.011744154797097556, 0.008519215264713786, 0.009083579682880946, 0.09639881752217147]
    # lbp1exudat = [0.03490698438293171, 0.01585595270088686, 0.0017020514198692106, 0.006001970796380901, 0.017408701364627188, 0.0020603780345785183, 0.0069276478843799455, 0.02245513452178327, 0.0012242826002568006, 0.00041804771716085875, 5.972110245155125e-05, 0.0003284660634835319, 0.006360297411090208, 0.0008659559855474932, 0.006479739615993312, 0.015228881125145569, 0.01803577294036848, 0.005792946937800473, 0.00041804771716085875, 0.0014034459076114547, 0.002388844098062051, 0.0003881871659350832, 0.0006270715757412882, 0.001702051419869211, 0.006987368986831496, 0.0015228881125145574, 0.0003284660634835319, 0.0003583266147093075, 0.025321747439457736, 0.0020603780345785183, 0.016154558213144612, 0.021439875780106903, 0.0013138642539341277, 0.00041804771716085875, 8.958165367732688e-05, 0.00014930275612887815, 0.0005972110245155126, 0.00014930275612887813, 0.0001194422049031025, 0.0007763743318701663, 8.958165367732688e-05, 0.0, 0.0, 2.9860551225775625e-05, 0.00038818716593508307, 0.0, 0.0002687449610319806, 0.00044790826838663443, 0.008122069933410971, 0.0016721908686434352, 0.0001194422049031025, 0.0002687449610319806, 0.0007465137806443906, 0.0001194422049031025, 0.0003881871659350831, 0.0006867926781928394, 0.010779658992505001, 0.0011048403953536983, 0.0002687449610319806, 0.00038818716593508307, 0.02908417689390546, 0.0011347009465794739, 0.011585893875600946, 0.009883842455731732, 0.018423960106303566, 0.0021499596882558454, 0.00041804771716085875, 0.0008062348830959418, 0.007226253396637701, 0.000238884409806205, 0.001851354175998089, 0.002209680790707396, 0.0006867926781928394, 0.00014930275612887815, 0.0, 2.9860551225775625e-05, 0.00143330645883723, 0.0, 0.0009853981904505958, 0.0010152587416763714, 0.002329122995610499, 0.0003583266147093075, 2.9860551225775625e-05, 0.0001194422049031025, 0.00029860551225775625, 2.9860551225775625e-05, 5.972110245155125e-05, 0.0001194422049031025, 0.0007763743318701664, 8.958165367732688e-05, 2.9860551225775625e-05, 5.972110245155125e-05, 0.0019409358296754156, 0.00023888440980620506, 0.0006867926781928394, 0.0010152587416763714, 0.007823464421153215, 0.0008659559855474932, 0.00017916330735465375, 0.0002986055122577563, 0.0015527486637403325, 0.0001194422049031025, 0.0004180477171608587, 0.0006270715757412881, 0.00038818716593508307, 2.9860551225775625e-05, 2.9860551225775625e-05, 2.9860551225775625e-05, 0.0007763743318701663, 8.958165367732688e-05, 0.000716653229418615, 0.00038818716593508307, 0.02696407775687539, 0.0020603780345785183, 0.0005374899220639613, 0.00044790826838663443, 0.002418704649287826, 8.958165367732688e-05, 0.0005076293708381857, 0.0006270715757412882, 0.022126668458299737, 0.0007763743318701663, 0.00032846606348353193, 0.00041804771716085875, 0.03622084863686584, 0.0010749798441279227, 0.014034459076114546, 0.0071963928454119265, 0.0015826092149661083, 0.005554062527994267, 8.958165367732688e-05, 0.006360297411090208, 0.00041804771716085875, 0.0007465137806443906, 0.0003881871659350831, 0.020573919794559404, 0.0, 5.972110245155125e-05, 0.0, 0.000238884409806205, 0.0002687449610319806, 0.00020902385858042938, 0.00017916330735465375, 0.010660216787601899, 0.0006270715757412882, 0.001104840395353698, 5.972110245155125e-05, 0.0010152587416763714, 0.0001194422049031025, 0.0, 0.0, 0.0007763743318701663, 0.00032846606348353193, 0.000238884409806205, 0.0, 0.00038818716593508307, 0.0002687449610319806, 0.0005972110245155126, 0.0002687449610319806, 0.010361611275344141, 0.0001194422049031025, 0.0002687449610319806, 0.0, 0.0003284660634835319, 2.9860551225775625e-05, 2.9860551225775625e-05, 2.9860551225775625e-05, 0.0004479082683866343, 0.0, 0.0, 0.0, 5.972110245155125e-05, 0.0, 0.0, 2.9860551225775625e-05, 0.0005972110245155126, 0.000238884409806205, 0.0006569321269670639, 0.0, 0.0003881871659350832, 2.9860551225775625e-05, 2.9860551225775625e-05, 2.9860551225775625e-05, 0.0005972110245155125, 0.0002687449610319806, 0.0005673504732897369, 2.9860551225775625e-05, 0.0009555376392248202, 0.0006569321269670639, 0.0005972110245155126, 0.0003881871659350831, 0.00922691032876467, 0.006599181820896413, 0.02275374003404103, 0.0002687449610319806, 0.02263429782913793, 0.0017319119710949864, 0.0017916330735465377, 0.000716653229418615, 0.028457105318164175, 0.00017916330735465375, 0.0005673504732897369, 0.0, 0.0006569321269670638, 0.00041804771716085875, 0.0005972110245155126, 0.0008659559855474932, 0.010630356236376126, 0.000716653229418615, 0.00212009913703007, 0.0, 0.00095553763922482, 0.00014930275612887813, 5.972110245155125e-05, 0.0, 0.0007763743318701663, 0.000238884409806205, 0.00041804771716085875, 0.0, 0.00044790826838663443, 0.00047776881961241, 0.0007763743318701661, 0.0004479082683866343, 0.006987368986831496, 0.008062348830959418, 0.018364239003852013, 0.00017916330735465375, 0.010749798441279226, 0.0008062348830959418, 0.0014034459076114544, 0.0006569321269670639, 0.010391471826569916, 0.0002986055122577563, 0.0005673504732897369, 5.972110245155125e-05, 0.0003881871659350831, 0.0006270715757412882, 0.0006569321269670639, 0.0008659559855474932, 0.008032488279733643, 0.02054405924333363, 0.02663561169339186, 0.0003284660634835319, 0.011048403953536982, 0.0009256770879990445, 0.001134700946579474, 0.00023888440980620506, 0.007076950640508823, 0.014243482934694974, 0.01146645167069784, 0.00029860551225775625, 0.009376213084893548, 0.012869897578309294, 0.008928304816506912, 0.009495655289796649, 0.10937919914001612]
    # procesarePatchuri(path_imagine1, image1_file, image1_file1, image1_file2, '*.png', vect1retina, vect1exudat, image1_file3, lbp1retina, lbp1exudat, "retina", 1)
    # image1_file.close()
    # image1_file1.close()
    # image1_file2.close()
    # image1_file3.close()

    vect1retina = [83.718, 107.19, 118.19, 1.099, 1.093, 1.093, 0.9243, 0.784, 0.7192]
    vect1exudat = [195.03, 199.483, 159.876, 1.068, 1.069, 1.05, 0.5235, 0.4433, 0.5304]
    lbp1pad_R = [0.194, 0.091, 0.104, 0.018, 0.099, 0.094, 0.016, 0.093, 0.035, 0.256]
    lbp1camp_R = [0.199, 0.076, 0.092, 0.018, 0.103, 0.11, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_G = [0.193, 0.092, 0.105, 0.018, 0.101, 0.095, 0.016, 0.092, 0.035, 0.254]
    lbp1camp_G = [0.199, 0.075, 0.091, 0.018, 0.103, 0.111, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_B = [0.196, 0.09, 0.104, 0.018, 0.096, 0.094, 0.016, 0.094, 0.036, 0.257]
    lbp1camp_B = [0.199, 0.075, 0.092, 0.018, 0.103, 0.109, 0.017, 0.099, 0.034, 0.254]
    image1_file = open("image1ret_intensitate.csv", "w")
    image1_file1 = open("image1ret_lacun.csv", "w")
    image1_file2 = open("image1ret_lbp.csv", "w")
    image1_file3 = open("rezultate_clasificare_imagine1ret.csv", "w")
    image1_file4 = open("image1ret_contrast.csv", "w")
    path_imagine1 = 'retina\split retina 1'
    procesarePatchuri(path_imagine1, image1_file, image1_file1, image1_file2, '*.png', vect1retina, vect1exudat,
                      image1_file3, lbp1pad_R, lbp1camp_R, lbp1pad_G, lbp1camp_G, lbp1pad_B, lbp1camp_B, "retina", 1,
                      image1_file4)
    image1_file.close()
    image1_file1.close()
    image1_file2.close()
    image1_file3.close()
    image1_file4.close()

    vect2retina = [83.718, 107.19, 118.19, 1.099, 1.093, 1.093, 0.9243, 0.784, 0.7192]
    vect2exudat = [195.03, 199.483, 159.876, 1.068, 1.069, 1.05, 0.5235, 0.4433, 0.5304]
    lbp1pad_R = [0.194, 0.091, 0.104, 0.018, 0.099, 0.094, 0.016, 0.093, 0.035, 0.256]
    lbp1camp_R = [0.199, 0.076, 0.092, 0.018, 0.103, 0.11, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_G = [0.193, 0.092, 0.105, 0.018, 0.101, 0.095, 0.016, 0.092, 0.035, 0.254]
    lbp1camp_G = [0.199, 0.075, 0.091, 0.018, 0.103, 0.111, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_B = [0.196, 0.09, 0.104, 0.018, 0.096, 0.094, 0.016, 0.094, 0.036, 0.257]
    lbp1camp_B = [0.199, 0.075, 0.092, 0.018, 0.103, 0.109, 0.017, 0.099, 0.034, 0.254]
    image2_file = open("image2ret_intensitate.csv", "w")
    image2_file1 = open("image2ret_lacun.csv", "w")
    image2_file2 = open("image2ret_lbp.csv", "w")
    image2_file3 = open("rezultate_clasificare_imagine2ret.csv", "w")
    image2_file4 = open("image2ret_contrast.csv", "w")
    path_imagine2 = 'retina\split retina 2'
    procesarePatchuri(path_imagine2, image2_file, image2_file1, image2_file2, '*.png', vect2retina, vect2exudat,
                      image2_file3, lbp1pad_R, lbp1camp_R, lbp1pad_G, lbp1camp_G, lbp1pad_B, lbp1camp_B, "retina", 2,
                      image2_file4)
    image2_file.close()
    image2_file1.close()
    image2_file2.close()
    image2_file3.close()
    image2_file4.close()

    vect3retina = [83.718, 107.19, 118.19, 1.099, 1.093, 1.093, 0.9243, 0.784, 0.7192]
    vect3exudat = [195.03, 199.483, 159.876, 1.068, 1.069, 1.05, 0.5235, 0.4433, 0.5304]
    lbp1pad_R = [0.194, 0.091, 0.104, 0.018, 0.099, 0.094, 0.016, 0.093, 0.035, 0.256]
    lbp1camp_R = [0.199, 0.076, 0.092, 0.018, 0.103, 0.11, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_G = [0.193, 0.092, 0.105, 0.018, 0.101, 0.095, 0.016, 0.092, 0.035, 0.254]
    lbp1camp_G = [0.199, 0.075, 0.091, 0.018, 0.103, 0.111, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_B = [0.196, 0.09, 0.104, 0.018, 0.096, 0.094, 0.016, 0.094, 0.036, 0.257]
    lbp1camp_B = [0.199, 0.075, 0.092, 0.018, 0.103, 0.109, 0.017, 0.099, 0.034, 0.254]
    image3_file = open("image3ret_intensitate.csv", "w")
    image3_file1 = open("image3ret_lacun.csv", "w")
    image3_file2 = open("image3ret_lbp.csv", "w")
    image3_file3 = open("rezultate_clasificare_imagine3ret.csv", "w")
    image3_file4 = open("image3ret_contrast.csv", "w")
    path_imagine3 = 'retina\split retina 3'
    procesarePatchuri(path_imagine3, image3_file, image3_file1, image3_file2, '*.png', vect3retina, vect3exudat,
                      image3_file3, lbp1pad_R, lbp1camp_R, lbp1pad_G, lbp1camp_G, lbp1pad_B, lbp1camp_B, "retina", 3,
                      image3_file4)
    image3_file.close()
    image3_file1.close()
    image3_file2.close()
    image3_file3.close()
    image3_file4.close()

    vect4retina = [83.718, 107.19, 118.19, 1.099, 1.093, 1.093, 0.9243, 0.784, 0.7192]
    vect4exudat = [195.03, 199.483, 159.876, 1.068, 1.069, 1.05, 0.5235, 0.4433, 0.5304]
    lbp1pad_R = [0.194, 0.091, 0.104, 0.018, 0.099, 0.094, 0.016, 0.093, 0.035, 0.256]
    lbp1camp_R = [0.199, 0.076, 0.092, 0.018, 0.103, 0.11, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_G = [0.193, 0.092, 0.105, 0.018, 0.101, 0.095, 0.016, 0.092, 0.035, 0.254]
    lbp1camp_G = [0.199, 0.075, 0.091, 0.018, 0.103, 0.111, 0.017, 0.1, 0.034, 0.252]
    lbp1pad_B = [0.196, 0.09, 0.104, 0.018, 0.096, 0.094, 0.016, 0.094, 0.036, 0.257]
    lbp1camp_B = [0.199, 0.075, 0.092, 0.018, 0.103, 0.109, 0.017, 0.099, 0.034, 0.254]
    image4_file = open("image4ret_intensitate.csv", "w")
    image4_file1 = open("image4ret_lacun.csv", "w")
    image4_file2 = open("image4ret_lbp.csv", "w")
    image4_file3 = open("rezultate_clasificare_imagine1ret.csv", "w")
    image4_file4 = open("image4ret_contrast.csv", "w")
    path_imagine4 = 'retina\split retina 4'
    procesarePatchuri(path_imagine4, image4_file, image4_file1, image4_file2, '*.png', vect4retina, vect4exudat,
                      image4_file3, lbp1pad_R, lbp1camp_R, lbp1pad_G, lbp1camp_G, lbp1pad_B, lbp1camp_B, "retina", 4,
                      image4_file4)
    image4_file.close()
    image4_file1.close()
    image4_file2.close()
    image4_file3.close()
    image4_file4.close()

    # image2_file = open("image2ret_df.txt", "a")
    # image2_file1 = open("image2ret_lacun.txt", "a")
    # image2_file2 = open("image2ret_lbp.csv", "a")
    # image2_file3 = open("clasificare_retina2.csv", "a")
    # path_imagine2 = 'retina\split retina 2'
    # vect2retina = [1.670694937, 1.670810638, 1.67118591, 1.128906653, 1.095061353, 1.041139354]
    # vect2exudat = [1.691765393, 1.69176291, 1.69203378, 1.046548854, 1.048391319, 1.029672289]
    # lbp2retina = [0.035635581832840636, 0.017334049986562752, 0.0010481053480247246, 0.006933619994625101, 0.014780972856758934, 0.002123085192152647, 0.006799247514109109, 0.02346143509809191, 0.0014780972856758934, 0.0004837409298575652, 0.00010749798441279225, 8.06234883095942e-05, 0.006906745498521903, 0.0007793603869927438, 0.009137328675087342, 0.02244020424617038, 0.01703843052942757, 0.007309862940069874, 0.0005106154259607633, 0.0013974737973662993, 0.0019349637194302609, 0.0003224939532383767, 0.0008868583714055361, 0.002123085192152647, 0.0074173609244826645, 0.001639344262295082, 0.0001612469766191884, 0.0006449879064767534, 0.02792260145122279, 0.002203708680462241, 0.03224939532383768, 0.02735823703305563, 0.0012631013168503092, 0.0005106154259607633, 8.06234883095942e-05, 0.00024187046492878255, 0.0002956194571351787, 0.00010749798441279225, 0.0003224939532383767, 0.0004837409298575651, 5.3748992206396127e-05, 2.6874496103198063e-05, 0.0, 2.6874496103198063e-05, 0.0001612469766191884, 0.0, 0.00024187046492878255, 0.0005912389142703575, 0.007067992475141091, 0.0015855952700886858, 0.0001343724805159903, 0.0005374899220639613, 0.0005106154259607634, 2.6874496103198063e-05, 0.0005106154259607633, 0.0003224939532383768, 0.008680462241332974, 0.0011824778285407147, 0.0002687449610319807, 0.0003224939532383767, 0.020155872077398548, 0.0006987368986831495, 0.013867239989250203, 0.011314162859446386, 0.014028486965869388, 0.0019349637194302609, 0.0003762429454447729, 0.000913732867508734, 0.005482397205052406, 0.00018812147272238643, 0.0013705993012631012, 0.001988712711636657, 0.00029561945713517867, 8.06234883095942e-05, 5.3748992206396127e-05, 2.6874496103198063e-05, 0.0017199677506046763, 0.0, 0.0013705993012631012, 0.0006987368986831495, 0.0017737167428110723, 0.00029561945713517867, 0.00013437248051599034, 0.0, 0.0003224939532383768, 5.3748992206396127e-05, 0.0001343724805159903, 5.3748992206396127e-05, 0.0005374899220639613, 0.0, 0.0, 0.0, 0.0019080892233270628, 2.6874496103198063e-05, 0.0007524858908895458, 0.0004837409298575652, 0.006906745498521903, 0.0011556033324375168, 0.0001612469766191884, 0.00037624294544477287, 0.0011018543402311206, 5.3748992206396127e-05, 0.00018812147272238643, 0.0005374899220639614, 8.06234883095942e-05, 0.0, 2.6874496103198063e-05, 5.3748992206396127e-05, 0.0008868583714055361, 5.3748992206396127e-05, 0.0006718624025799515, 0.0004568664337543671, 0.025880139747379738, 0.002149959688255845, 0.0003493684493415748, 0.0006449879064767535, 0.0018005912389142704, 2.6874496103198063e-05, 0.0005912389142703575, 0.0005374899220639614, 0.024966406879871, 0.0008868583714055362, 0.000456866433754367, 0.0003224939532383767, 0.02652512765385649, 0.0006987368986831496, 0.011771029293200754, 0.0069067454985219014, 0.0009943563558183284, 0.007551733404998656, 2.6874496103198063e-05, 0.008707336737436174, 0.0004837409298575651, 0.00040311744154797097, 0.0003224939532383768, 0.0208546089760817, 2.6874496103198063e-05, 0.0001343724805159903, 0.0, 0.0002149959688255845, 0.0002149959688255845, 0.0002149959688255845, 0.00018812147272238643, 0.010722923945176027, 0.0005374899220639614, 0.001289975812953507, 0.0, 0.0009137328675087343, 0.00010749798441279225, 8.06234883095942e-05, 0.0, 0.0006181134103735553, 0.0001343724805159903, 0.0004837409298575652, 0.0, 0.0002687449610319807, 0.0004837409298575651, 0.0006987368986831497, 0.00037624294544477287, 0.010454178984144047, 5.3748992206396127e-05, 0.0001343724805159903, 0.0, 0.00024187046492878255, 5.3748992206396127e-05, 5.3748992206396127e-05, 0.0, 0.0003224939532383767, 2.6874496103198063e-05, 2.6874496103198063e-05, 0.0, 0.0, 0.0, 2.6874496103198063e-05, 2.6874496103198063e-05, 0.00029561945713517867, 0.00021499596882558453, 0.0006987368986831496, 0.0, 0.00037624294544477287, 5.3748992206396127e-05, 5.3748992206396127e-05, 2.6874496103198063e-05, 0.0005106154259607633, 0.00016124697661918836, 0.0002687449610319806, 0.0, 0.0010481053480247246, 0.0005912389142703575, 0.000429991937651169, 0.0007256113947863476, 0.008384842784197797, 0.006745498521902714, 0.025692018274657353, 0.0002149959688255845, 0.022870196183821554, 0.0011556033324375168, 0.0015049717817790915, 0.0006718624025799518, 0.025960763235689327, 8.06234883095942e-05, 0.0003493684493415748, 0.0, 0.0004837409298575652, 0.00024187046492878255, 0.0004837409298575652, 0.0004568664337543671, 0.01152915882827197, 0.0006181134103735556, 0.0021499596882558454, 5.3748992206396127e-05, 0.0005912389142703575, 2.6874496103198063e-05, 8.06234883095942e-05, 2.6874496103198063e-05, 0.0006181134103735554, 0.0001612469766191884, 0.0004568664337543671, 2.6874496103198063e-05, 0.0004837409298575651, 0.0005106154259607633, 0.0007256113947863476, 0.0003224939532383768, 0.005993012631013168, 0.010024187046492878, 0.032625638269282456, 0.0001343724805159903, 0.012496640687987098, 0.0009674818597151304, 0.0007524858908895457, 0.0004837409298575652, 0.011179790378930394, 0.0005374899220639614, 0.0003224939532383768, 0.0, 0.0005912389142703575, 0.00040311744154797097, 0.0005374899220639613, 0.0011556033324375166, 0.010830421929588819, 0.023407686105885513, 0.029508196721311476, 0.0005912389142703575, 0.010158559527008868, 0.00083310937919914, 0.0009137328675087343, 0.0003493684493415748, 0.006342381080354743, 0.011582907820478366, 0.011179790378930396, 0.0005643644181671595, 0.008384842784197797, 0.01222789572695512, 0.007336737436173071, 0.010507927976350443, 0.08298844396667562]
    # lbp2exudat = [0.0333243751679656, 0.02058586401504972, 0.0008062348830959419, 0.005804891158290782, 0.014995968825584521, 0.001988712711636657, 0.0065036280569739325, 0.027250739048642836, 0.000429991937651169, 0.000429991937651169, 0.0002149959688255845, 0.000429991937651169, 0.005804891158290783, 0.0007524858908895457, 0.009621069604944906, 0.0318731523783929, 0.019403386186509004, 0.010803547433485623, 0.00037624294544477287, 0.0020962106960494492, 0.002096210696049449, 0.0004837409298575651, 0.0005374899220639613, 0.0026874496103198066, 0.007686105885514648, 0.0015049717817790915, 0.0003224939532383767, 0.0004837409298575651, 0.023810803547433486, 0.0023112066648750336, 0.021822090835796826, 0.03692555764579414, 0.0007524858908895457, 0.0004837409298575651, 0.0, 0.00010749798441279225, 0.0003224939532383768, 0.0001612469766191884, 0.00010749798441279225, 0.0005912389142703575, 0.0, 5.3748992206396127e-05, 0.0, 0.0, 0.0001612469766191884, 0.0, 0.0005912389142703573, 0.0005374899220639614, 0.005858640150497178, 0.0020962106960494492, 5.3748992206396127e-05, 0.000429991937651169, 0.0004837409298575651, 0.00010749798441279225, 0.0002149959688255845, 0.0004837409298575651, 0.0072561139478634785, 0.001343724805159903, 0.0002149959688255845, 0.0003224939532383768, 0.013974737973662993, 0.0004837409298575651, 0.009674818597151304, 0.009836065573770491, 0.015587207739854878, 0.001988712711636657, 0.0003224939532383768, 0.0006449879064767535, 0.004407417360924482, 0.0003224939532383768, 0.001343724805159903, 0.0017737167428110723, 0.00037624294544477287, 0.0, 0.0, 0.0, 0.0009674818597151302, 5.3748992206396127e-05, 0.0014512227895726953, 0.0005374899220639614, 0.0014512227895726953, 0.0005374899220639613, 0.00010749798441279225, 0.0, 0.00010749798441279225, 5.3748992206396127e-05, 0.00010749798441279225, 0.00010749798441279225, 0.0004837409298575651, 5.3748992206396127e-05, 0.0, 0.0, 0.0021499596882558454, 5.3748992206396127e-05, 0.000859983875302338, 0.0007524858908895457, 0.005374899220639613, 0.0007524858908895457, 0.0001612469766191884, 0.0002687449610319806, 0.0009674818597151302, 5.3748992206396127e-05, 0.0002149959688255845, 0.00037624294544477287, 0.0, 0.0, 0.0, 0.0, 0.0010212308519215265, 0.0, 0.000429991937651169, 0.00037624294544477287, 0.026713249126578874, 0.0018274657350174685, 0.0005374899220639613, 0.0008062348830959419, 0.0018812147272238642, 0.00010749798441279225, 0.0007524858908895457, 0.0008062348830959418, 0.024885783391561406, 0.0006987368986831495, 0.0007524858908895458, 0.000429991937651169, 0.022897070679924754, 0.0007524858908895457, 0.009943563558183283, 0.008976081698468153, 0.0011824778285407151, 0.007524858908895457, 5.3748992206396127e-05, 0.007739854877721042, 0.00037624294544477287, 0.0005374899220639613, 0.00010749798441279225, 0.027895726955119594, 0.0001612469766191884, 0.0, 0.0, 0.0001612469766191884, 0.0002687449610319806, 0.0002149959688255845, 0.0003224939532383768, 0.013490997043805427, 0.0006449879064767535, 0.002042461703843053, 0.0001612469766191884, 0.0010749798441279227, 0.0, 5.3748992206396127e-05, 0.0, 0.000859983875302338, 0.0002687449610319807, 0.0003224939532383768, 0.0, 0.0005912389142703573, 0.0002687449610319806, 0.000859983875302338, 0.0003224939532383768, 0.01166353130878796, 5.3748992206396127e-05, 0.0002149959688255845, 0.0, 0.0001612469766191884, 0.0, 5.3748992206396127e-05, 0.0, 0.00037624294544477287, 0.0, 0.0, 5.3748992206396127e-05, 0.0, 0.0, 5.3748992206396127e-05, 0.00010749798441279225, 0.0003224939532383768, 0.0003224939532383767, 0.0009674818597151304, 0.0, 0.0006449879064767535, 0.0, 0.0, 0.0, 0.0006449879064767535, 0.0002149959688255845, 0.000429991937651169, 0.0, 0.0008062348830959419, 0.0002687449610319806, 0.0001612469766191884, 0.0002687449610319807, 0.007847352862133835, 0.007202364955657081, 0.019080892233270624, 0.0001612469766191884, 0.012362268207471109, 0.0009674818597151302, 0.002203708680462241, 0.0008062348830959418, 0.024670787422735825, 5.3748992206396127e-05, 0.0005912389142703575, 0.0, 0.00037624294544477287, 0.000429991937651169, 0.0005912389142703573, 0.00037624294544477287, 0.010534802472453642, 0.0008062348830959419, 0.0023112066648750336, 5.3748992206396127e-05, 0.0009137328675087342, 5.3748992206396127e-05, 0.00010749798441279225, 0.0, 0.0008062348830959419, 0.0001612469766191884, 0.0006449879064767534, 0.0, 0.0001612469766191884, 0.0004837409298575651, 0.0010212308519215265, 0.0006449879064767535, 0.007417360924482666, 0.012147272238645524, 0.023273313625369524, 0.0002687449610319807, 0.00859983875302338, 0.0007524858908895457, 0.0007524858908895457, 0.0002149959688255845, 0.010051061542596076, 0.00042999193765116907, 0.0003224939532383768, 5.3748992206396127e-05, 0.0002687449610319806, 0.0005912389142703573, 0.0002687449610319807, 0.0012362268207471109, 0.010964794410104813, 0.03585057780166622, 0.03735554958344531, 0.0002687449610319806, 0.010857296425692017, 0.000913732867508734, 0.0008062348830959419, 0.0005912389142703573, 0.006503628056973932, 0.011932276269819941, 0.010534802472453642, 0.0003224939532383768, 0.007793603869927439, 0.010534802472453642, 0.006557377049180327, 0.011932276269819941, 0.0876646062886321]
    # procesarePatchuri(path_imagine2, image2_file, image2_file1, image2_file2, '*.png', vect2retina, vect2exudat, image2_file3, lbp2retina, lbp2exudat, "retina", 2)
    # image2_file.close()
    # image2_file1.close()
    # image2_file2.close()
    # image2_file3.close()

    # image3_file = open("image3ret_df.txt", "a")
    # image3_file1 = open("image3ret_lacun.txt", "a")
    # image3_file2 = open("image3ret_lbp.csv", "a")
    # image3_file3 = open("clasificare_retina3.csv", "a")
    # path_imagine3 = 'retina\split retina 3'
    # vect3retina = [1.703502168, 1.703617791, 1.703452502, 1.056274235, 1.036061573, 1.055451259]
    # vect3exudat = [1.703423506, 1.703427912, 1.703302729, 1.035190621, 1.03181862, 1.027405609]
    # lbp3retina = [0.03168503090567052, 0.022252082773447996, 0.001128728836334319, 0.005563020693361999, 0.016044074173609247, 0.0028218220908357967, 0.004810534802472453, 0.026686374630475677, 0.0010212308519215265, 0.00048374092985756527, 0.00010749798441279225, 0.0001343724805159903, 0.004864283794678849, 0.0007256113947863477, 0.007148615963450684, 0.02101585595270089, 0.02201021230851922, 0.010400429991937651, 0.0005643644181671594, 0.0021230851921526473, 0.002633700618113411, 0.0006181134103735556, 0.0009406073636119323, 0.0028486965869389948, 0.005643644181671593, 0.0016662187583982798, 0.0002687449610319807, 0.0007524858908895457, 0.022708949207202362, 0.0026874496103198066, 0.01625907014243483, 0.033754367105616766, 0.001209352324643913, 0.0008062348830959419, 2.6874496103198063e-05, 0.00018812147272238643, 0.00029561945713517867, 0.00010749798441279225, 0.0002149959688255845, 0.0006181134103735554, 0.0001343724805159903, 2.6874496103198063e-05, 0.0, 2.6874496103198063e-05, 0.0002149959688255845, 5.3748992206396127e-05, 0.00034936844934157477, 0.0004837409298575652, 0.006019887127116367, 0.001988712711636657, 0.0002149959688255845, 0.0003224939532383767, 0.0008868583714055362, 8.06234883095942e-05, 0.0003493684493415748, 0.0006181134103735556, 0.005079279763504434, 0.0013705993012631014, 0.00010749798441279225, 0.0003493684493415748, 0.01220102123085192, 0.0010212308519215265, 0.00806234883095942, 0.011582907820478366, 0.01655468959957001, 0.002660575114216609, 0.0005374899220639614, 0.0006449879064767535, 0.005482397205052406, 0.0007524858908895458, 0.0013437248051599033, 0.002069336199946251, 0.0004837409298575652, 0.00013437248051599034, 0.0, 0.0, 0.0014243482934694974, 5.3748992206396127e-05, 0.0009943563558183284, 0.0008331093791991399, 0.003251814028486966, 0.0007793603869927438, 2.6874496103198063e-05, 5.3748992206396127e-05, 0.0005912389142703573, 0.00024187046492878255, 2.6874496103198063e-05, 0.00018812147272238643, 0.0005374899220639613, 8.06234883095942e-05, 2.6874496103198063e-05, 5.3748992206396127e-05, 0.002123085192152647, 0.00010749798441279225, 0.0007256113947863477, 0.0010212308519215265, 0.00666487503359312, 0.0006449879064767537, 8.06234883095942e-05, 0.00013437248051599034, 0.001236226820747111, 8.06234883095942e-05, 0.00013437248051599034, 0.0007256113947863476, 0.00024187046492878255, 0.0, 0.0, 0.0, 0.00083310937919914, 0.0, 0.0005106154259607633, 0.00037624294544477287, 0.027035743079817254, 0.0028486965869389943, 0.0004568664337543671, 0.0011287288363343187, 0.002096210696049449, 0.0001343724805159903, 0.0006987368986831497, 0.0011824778285407147, 0.018435904326793873, 0.0011018543402311208, 0.0001612469766191884, 0.0003224939532383767, 0.02571889277076055, 0.0012093523246439132, 0.011878527277613547, 0.008411717280300995, 0.001289975812953507, 0.006772373018005912, 8.06234883095942e-05, 0.005670518677774792, 0.00037624294544477287, 0.0009406073636119323, 0.00018812147272238643, 0.015667831228164474, 5.3748992206396127e-05, 0.0002149959688255845, 0.0, 0.0002687449610319806, 5.3748992206396127e-05, 0.00024187046492878255, 0.0002687449610319807, 0.010669174952969632, 0.000456866433754367, 0.00236495565708143, 8.06234883095942e-05, 0.0008868583714055361, 2.6874496103198063e-05, 0.0001343724805159903, 0.0, 0.0010212308519215265, 0.00018812147272238643, 0.00018812147272238643, 0.0, 0.00037624294544477287, 0.0004837409298575651, 0.0005912389142703576, 0.00040311744154797097, 0.01155603332437517, 0.0001612469766191884, 0.0002687449610319806, 0.0, 0.0002149959688255845, 2.6874496103198063e-05, 2.6874496103198063e-05, 2.6874496103198063e-05, 0.00040311744154797097, 0.0, 2.6874496103198063e-05, 0.0, 0.0, 2.6874496103198063e-05, 5.3748992206396127e-05, 0.0, 0.00040311744154797097, 0.00029561945713517867, 0.0007256113947863477, 0.0, 0.0003493684493415749, 5.3748992206396127e-05, 2.6874496103198063e-05, 2.6874496103198063e-05, 0.00034936844934157477, 0.00010749798441279225, 0.00042999193765116907, 2.6874496103198063e-05, 0.00083310937919914, 0.0002149959688255845, 0.0002149959688255845, 0.00029561945713517867, 0.00792797635044343, 0.005374899220639613, 0.025262026337006183, 0.0003224939532383768, 0.013867239989250203, 0.0016124697661918839, 0.0019349637194302609, 0.0006449879064767534, 0.026041386723998926, 5.3748992206396127e-05, 0.00037624294544477287, 2.6874496103198063e-05, 0.00029561945713517867, 0.00034936844934157477, 0.0005374899220639614, 0.00045686643375436717, 0.01179790378930395, 0.0007524858908895457, 0.0033055630206933625, 0.0, 0.00083310937919914, 0.00010749798441279225, 0.0001343724805159903, 5.3748992206396127e-05, 0.0012899758129535069, 0.00018812147272238643, 0.0007524858908895457, 2.6874496103198063e-05, 0.00018812147272238643, 0.0008868583714055361, 0.001289975812953507, 0.0004837409298575651, 0.00929857565170653, 0.00819672131147541, 0.026552002149959692, 0.00018812147272238643, 0.009621069604944908, 0.0009943563558183284, 0.0013705993012631012, 0.00037624294544477287, 0.012335393711367913, 0.00029561945713517867, 0.0005643644181671594, 2.6874496103198063e-05, 0.00042999193765116907, 0.0005374899220639613, 0.0005106154259607633, 0.0014243482934694974, 0.010642300456866432, 0.02819134641225477, 0.04047299113141629, 0.00040311744154797097, 0.01222789572695512, 0.001209352324643913, 0.001182477828540715, 0.0006181134103735554, 0.008787960225745768, 0.011394786347755979, 0.012523515184090301, 0.0003493684493415748, 0.008196721311475409, 0.012281644719161516, 0.009406073636119321, 0.010454178984144049, 0.09897876914807849]
    # lbp3exudat = [0.02889008331093792, 0.01599032518140285, 0.0014109110454178983, 0.007860790110185435, 0.016191883902176834, 0.0017468422467078742, 0.006315506584251546, 0.02929320075248589, 0.000873421123353937, 0.0006046761623219564, 0.0, 0.0, 0.005374899220639613, 0.0007390486428379467, 0.008801397473797367, 0.023246439129266327, 0.016460628863208815, 0.005845202902445579, 0.0006718624025799516, 0.0008734211233539372, 0.0018812147272238647, 0.0002687449610319806, 0.000873421123353937, 0.0018812147272238644, 0.005912389142703574, 0.001276538564901908, 0.0001343724805159903, 0.000873421123353937, 0.025732330018812147, 0.002889008331093792, 0.019618382155334586, 0.036818059661381346, 0.0010749798441279225, 0.00033593120128997577, 0.0, 0.0, 0.00040311744154797097, 6.718624025799516e-05, 0.0001343724805159903, 0.0004031174415479709, 6.718624025799516e-05, 0.0001343724805159903, 0.0, 0.0, 0.0001343724805159903, 0.0, 0.00033593120128997577, 0.0005374899220639614, 0.005576457941413598, 0.0011421660843859178, 0.00020155872077398548, 6.718624025799516e-05, 0.0006718624025799515, 0.0, 0.0003359312012899758, 0.0006718624025799515, 0.008465466272507391, 0.0008062348830959419, 0.00020155872077398548, 0.0004703036818059661, 0.015184090298306908, 0.0008062348830959418, 0.00913732867508734, 0.011085729642569202, 0.01914807847352862, 0.0021499596882558454, 0.00040311744154797097, 0.0006046761623219564, 0.005509271701155603, 0.00040311744154797097, 0.001343724805159903, 0.0016124697661918839, 0.00033593120128997577, 0.0001343724805159903, 0.0, 0.0, 0.0019484009674818597, 6.718624025799516e-05, 0.001276538564901908, 0.0009406073636119322, 0.0022171459285138403, 0.00040311744154797097, 0.0, 0.0, 0.0003359312012899758, 6.718624025799516e-05, 0.0, 0.0001343724805159903, 0.0007390486428379468, 6.718624025799516e-05, 0.0, 0.00020155872077398548, 0.0023515184090298308, 6.718624025799516e-05, 0.0009406073636119322, 0.0009406073636119322, 0.00611394786347756, 0.0008734211233539371, 0.0001343724805159903, 0.00020155872077398548, 0.0010077936038699275, 0.0, 0.0001343724805159903, 0.0002687449610319806, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0008062348830959419, 0.0, 0.0001343724805159903, 0.0006046761623219564, 0.027277613544746036, 0.0020827734479978498, 0.00020155872077398548, 0.0009406073636119322, 0.0028218220908357967, 0.0001343724805159903, 0.0004703036818059661, 0.0008734211233539372, 0.02096210696049449, 0.0006046761623219564, 0.00040311744154797097, 0.0002687449610319806, 0.028957269551195916, 0.0008734211233539372, 0.011354474603601183, 0.007793603869927439, 0.0010749798441279225, 0.0069873689868314975, 6.718624025799516e-05, 0.010682612201021232, 0.00033593120128997577, 0.0009406073636119322, 0.00020155872077398548, 0.025396398817522172, 0.0, 0.0002687449610319806, 0.0, 0.00020155872077398548, 0.0002687449610319806, 0.0002687449610319806, 0.0001343724805159903, 0.012966944369793065, 0.0002687449610319806, 0.0014109110454178985, 0.0, 0.0015452835259338886, 6.718624025799516e-05, 0.0001343724805159903, 6.718624025799516e-05, 0.0010077936038699275, 6.718624025799516e-05, 0.00020155872077398548, 0.0, 0.0003359312012899758, 0.0006718624025799516, 0.0006046761623219564, 0.0001343724805159903, 0.01169040580489116, 0.0002687449610319806, 0.00020155872077398548, 6.718624025799516e-05, 6.718624025799516e-05, 0.0, 0.0001343724805159903, 0.0, 0.0003359312012899758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00033593120128997577, 0.0003359312012899758, 0.0010749798441279225, 0.0, 0.0006718624025799515, 0.0, 0.0, 0.0, 0.0004703036818059661, 0.0001343724805159903, 0.0003359312012899758, 0.0, 0.0008062348830959419, 0.0004703036818059661, 0.0003359312012899758, 0.0002687449610319806, 0.00859983875302338, 0.007121741467347487, 0.028957269551195916, 0.0005374899220639613, 0.021633969363074445, 0.0015452835259338886, 0.0024187046492878256, 0.0006718624025799515, 0.034533727492609516, 0.0, 0.0005374899220639613, 0.0, 0.0002687449610319806, 0.00020155872077398548, 0.0006046761623219564, 0.000806234883095942, 0.011421660843859179, 0.00040311744154797097, 0.0032249395323837677, 0.0, 0.0008062348830959418, 6.718624025799516e-05, 0.00020155872077398548, 6.718624025799516e-05, 0.0011421660843859178, 0.00020155872077398548, 0.0005374899220639613, 0.0001343724805159903, 0.0003359312012899758, 0.00033593120128997577, 0.0006046761623219564, 0.0003359312012899758, 0.01001074979844128, 0.009002956194571352, 0.020760548239720508, 0.0001343724805159903, 0.014310669174952969, 0.0006718624025799516, 0.0006046761623219564, 0.0005374899220639613, 0.01363880677237302, 0.0002687449610319806, 0.0001343724805159903, 0.0, 0.00020155872077398548, 0.00020155872077398548, 0.0004031174415479709, 0.0010749798441279227, 0.009271701155603333, 0.022037086804622415, 0.034063423810803545, 0.0001343724805159903, 0.011085729642569203, 0.0006046761623219565, 0.0016124697661918839, 0.0004031174415479709, 0.007793603869927439, 0.011287288363343188, 0.012227895726955118, 0.0004703036818059661, 0.008532652512765386, 0.009876377317925289, 0.007323300188121472, 0.0072561139478634785, 0.08290782047836603]
    # procesarePatchuri(path_imagine3, image3_file, image3_file1, image3_file2, '*.png', vect3retina, vect3exudat, image3_file3, lbp3retina, lbp3exudat, "retina", 3)
    # image3_file.close()
    # image3_file1.close()
    # image3_file2.close()
    # image3_file3.close()
    #
    # image4_file = open("image4ret_df.txt", "a")
    # image4_file1 = open("image4ret_lacun.txt", "a")
    # image4_file2 = open("image4ret_lbp.csv", "a")
    # image4_file3 = open("clasificare_retina4.csv", "a")
    # path_imagine4 = 'retina\split retina 4'
    # vect4retina = [1.679817917, 1.679804081, 1.679906526, 1.122925265, 1.105073453, 1.062902584]
    # vect4exudat = [1.686389932, 1.686389932, 1.686389932, 1.066072809, 1.108365568, 1.01611053]
    # lbp4retina = [0.02325865480931323, 0.01749285382717256, 0.0011482739244093722, 0.005716938261953044, 0.013266228530942316, 0.002003371527692947, 0.0053016051403581635, 0.03163861132149227, 0.000806234883095942, 0.00029317632112579704, 9.772544037526568e-05, 0.0002443136009381642, 0.004690821138012753, 0.0005619212821577777, 0.0069385062666438655, 0.02391830153184628, 0.01654003078351372, 0.008160074271334685, 0.00029317632112579704, 0.0013192934450660869, 0.001856783367130048, 0.00043976448168869564, 0.0007085094427206763, 0.002760743690601256, 0.003933448975104444, 0.0011482739244093718, 0.00017101952065671494, 0.0006107840023454106, 0.02240355720602966, 0.0026385868901321733, 0.018861009992426278, 0.03828394126701033, 0.001099411204221739, 0.00036647040140724634, 0.0, 4.886272018763284e-05, 0.0003664704014072463, 0.0, 9.772544037526568e-05, 0.000464195841782512, 4.886272018763284e-05, 0.0, 0.0, 0.0, 9.772544037526568e-05, 0.0, 0.00017101952065671494, 0.0004153331215948792, 0.0052527424201705315, 0.0009772544037526568, 4.886272018763284e-05, 0.0002443136009381642, 0.0005374899220639614, 7.329408028144926e-05, 0.00017101952065671494, 0.0004153331215948792, 0.005594781461483961, 0.0006352153624392269, 9.772544037526568e-05, 0.00021988224084434777, 0.013901443893381547, 0.0007329408028144927, 0.007500427548801642, 0.010114583078839999, 0.014805404216852754, 0.002369841929100193, 0.00043976448168869554, 0.0003909017615010627, 0.005765800982140676, 0.00029317632112579704, 0.0013192934450660867, 0.0019300774474114975, 0.00017101952065671494, 4.886272018763284e-05, 0.0, 0.0, 0.0011971366445970046, 4.886272018763284e-05, 0.0007085094427206763, 0.000928391683565024, 0.0018323520070362316, 0.00021988224084434777, 2.443136009381642e-05, 0.00014658816056289852, 0.0003176076812196135, 0.0, 2.443136009381642e-05, 9.772544037526568e-05, 0.0002687449610319806, 2.443136009381642e-05, 0.0, 0.0001221568004690821, 0.0019300774474114975, 0.00014658816056289852, 0.0004886272018763284, 0.0009283916835650242, 0.004519801617356037, 0.0004886272018763284, 7.329408028144926e-05, 0.0002687449610319806, 0.0011482739244093718, 4.886272018763284e-05, 0.00017101952065671494, 0.00043976448168869564, 0.00014658816056289852, 0.0, 0.0, 0.0, 0.00043976448168869564, 0.0, 0.00021988224084434777, 0.00017101952065671494, 0.029610808433705504, 0.0026874496103198066, 0.0003909017615010627, 0.0005619212821577778, 0.0022721164887249275, 4.886272018763284e-05, 0.00029317632112579704, 0.0008795289633773913, 0.02369841929100193, 0.0007085094427206762, 0.00017101952065671494, 0.0002687449610319806, 0.03615841293884831, 0.0010016857638464733, 0.009112897314993528, 0.007695878429552172, 0.0015391756859104348, 0.006058977303266474, 2.443136009381642e-05, 0.007182819867582028, 0.00043976448168869554, 0.0006107840023454106, 0.0002443136009381642, 0.022305831765654396, 2.443136009381642e-05, 7.329408028144926e-05, 2.443136009381642e-05, 9.772544037526568e-05, 7.329408028144926e-05, 0.0001221568004690821, 0.0001221568004690821, 0.011727052845031881, 0.0005374899220639613, 0.0014903129657228018, 4.886272018763284e-05, 0.000928391683565024, 7.329408028144926e-05, 0.00017101952065671494, 2.443136009381642e-05, 0.0009772544037526568, 0.00017101952065671494, 0.00014658816056289852, 0.0, 0.00017101952065671494, 0.0003420390413134299, 0.0003420390413134299, 0.0003909017615010627, 0.012069091886345314, 2.443136009381642e-05, 9.772544037526568e-05, 0.0, 0.00017101952065671494, 7.329408028144926e-05, 0.0, 0.0, 0.00036647040140724634, 0.0, 0.0, 0.0, 0.0, 2.443136009381642e-05, 0.0, 0.0, 0.00021988224084434782, 0.0001221568004690821, 0.0005619212821577777, 0.0, 0.0002443136009381642, 2.443136009381642e-05, 2.443136009381642e-05, 2.443136009381642e-05, 0.0003909017615010628, 7.329408028144926e-05, 0.00021988224084434777, 0.0, 0.0004153331215948792, 0.00019545088075053135, 0.0002687449610319806, 0.0002443136009381642, 0.005448193300921062, 0.006620898585424251, 0.032591434365151106, 0.00014658816056289852, 0.018128069189611785, 0.0014903129657228018, 0.002027802887786763, 0.0006596467225330433, 0.03833280398719797, 2.443136009381642e-05, 0.0003176076812196135, 0.0, 0.00017101952065671494, 7.329408028144926e-05, 0.00043976448168869554, 0.0002443136009381642, 0.011433876523906088, 0.0006596467225330433, 0.002882900491070338, 7.329408028144926e-05, 0.0010016857638464733, 9.772544037526568e-05, 0.0002443136009381642, 2.443136009381642e-05, 0.0014170188854413528, 0.00014658816056289852, 0.0005374899220639613, 2.443136009381642e-05, 0.0002687449610319806, 0.0003909017615010627, 0.0007818035230021254, 0.0003420390413134299, 0.01250885636803401, 0.00806234883095942, 0.02560406537831961, 0.00014658816056289852, 0.009894700837995651, 0.0008062348830959419, 0.0008306662431897583, 0.00043976448168869554, 0.013144071730473238, 7.329408028144926e-05, 0.00017101952065671494, 2.443136009381642e-05, 0.0002687449610319806, 0.00019545088075053135, 0.00031760768121961346, 0.0007085094427206763, 0.007158388507488212, 0.0235029684102514, 0.04761672082284821, 0.00031760768121961346, 0.012020229166157682, 0.0014170188854413525, 0.0013192934450660869, 0.00034203904131342993, 0.013486110771786665, 0.010163445799027632, 0.01011458307884, 0.00014658816056289852, 0.006058977303266474, 0.011433876523906084, 0.011116268842686473, 0.007158388507488212, 0.09616183332926144]
    # lbp4exudat = [0.029830690674549853, 0.018543402311206665, 0.0014780972856758936, 0.0064498790647675355, 0.011152915882827196, 0.001343724805159903, 0.005643644181671594, 0.02378392905133029, 0.0008062348830959419, 0.00040311744154797097, 0.0001343724805159903, 0.0001343724805159903, 0.005912389142703574, 0.0009406073636119322, 0.01155603332437517, 0.04823972050524053, 0.020155872077398548, 0.011152915882827196, 0.0005374899220639613, 0.0018812147272238644, 0.0018812147272238647, 0.0008062348830959419, 0.0001343724805159903, 0.0016124697661918839, 0.0041655468959957, 0.0013437248051599033, 0.0001343724805159903, 0.0002687449610319806, 0.013437248051599033, 0.0022843321687718355, 0.01706530502553077, 0.043671056167696856, 0.0032249395323837677, 0.001209352324643913, 0.0001343724805159903, 0.0, 0.0002687449610319806, 0.0, 0.0, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0, 0.0001343724805159903, 0.0, 0.0001343724805159903, 0.0002687449610319806, 0.005778016662187584, 0.0021499596882558454, 0.0001343724805159903, 0.0, 0.0002687449610319806, 0.0001343724805159903, 0.0001343724805159903, 0.0005374899220639613, 0.006584251545283526, 0.001343724805159903, 0.0002687449610319806, 0.0005374899220639613, 0.009540446116635314, 0.00040311744154797097, 0.004568664337543671, 0.0067186240257995165, 0.009674818597151304, 0.0018812147272238644, 0.0002687449610319806, 0.0005374899220639613, 0.0041655468959957, 0.0002687449610319806, 0.0008062348830959419, 0.001343724805159903, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0017468422467078742, 0.0001343724805159903, 0.0005374899220639613, 0.0005374899220639613, 0.0016124697661918839, 0.0005374899220639613, 0.0, 0.0, 0.0002687449610319806, 0.0001343724805159903, 0.0001343724805159903, 0.0, 0.0005374899220639613, 0.0002687449610319806, 0.0, 0.0, 0.0012093523246439128, 0.0001343724805159903, 0.0005374899220639613, 0.0006718624025799516, 0.00846546627250739, 0.0009406073636119322, 0.0001343724805159903, 0.0001343724805159903, 0.0009406073636119322, 0.0, 0.0002687449610319806, 0.0002687449610319806, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0006718624025799515, 0.0001343724805159903, 0.0002687449610319806, 0.0005374899220639613, 0.028890083310937918, 0.0026874496103198066, 0.0002687449610319806, 0.0001343724805159903, 0.002015587207739855, 0.0002687449610319806, 0.0001343724805159903, 0.0009406073636119322, 0.017602794947594733, 0.00040311744154797097, 0.00040311744154797097, 0.0002687449610319806, 0.022574576726686377, 0.0009406073636119322, 0.0061811341037355545, 0.0061811341037355545, 0.0012093523246439128, 0.006046761623219564, 0.0, 0.005509271701155604, 0.0006718624025799516, 0.0012093523246439128, 0.0, 0.014915345337274927, 0.0, 0.00040311744154797097, 0.0, 0.0001343724805159903, 0.0, 0.0, 0.0002687449610319806, 0.011287288363343188, 0.0009406073636119322, 0.0021499596882558454, 0.0, 0.0006718624025799516, 0.0001343724805159903, 0.0, 0.0, 0.00040311744154797097, 0.0, 0.00040311744154797097, 0.0, 0.0005374899220639613, 0.0001343724805159903, 0.00040311744154797097, 0.0002687449610319806, 0.008868583714055361, 0.0, 0.0005374899220639613, 0.0, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0005374899220639613, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001343724805159903, 0.0008062348830959419, 0.0, 0.0001343724805159903, 0.0, 0.0, 0.0, 0.0001343724805159903, 0.0, 0.0002687449610319806, 0.0, 0.0005374899220639613, 0.0002687449610319806, 0.00040311744154797097, 0.0001343724805159903, 0.006315506584251545, 0.005374899220639613, 0.017871539908626714, 0.0, 0.01195915076592314, 0.0002687449610319806, 0.0009406073636119322, 0.0009406073636119322, 0.021365224402042464, 0.0001343724805159903, 0.0001343724805159903, 0.0, 0.0002687449610319806, 0.0001343724805159903, 0.0, 0.0006718624025799516, 0.009137328675087342, 0.0012093523246439128, 0.0034936844934157487, 0.0, 0.0005374899220639613, 0.0001343724805159903, 0.0, 0.0, 0.00040311744154797097, 0.0002687449610319806, 0.0001343724805159903, 0.0, 0.00040311744154797097, 0.0005374899220639613, 0.0005374899220639613, 0.0001343724805159903, 0.006181134103735555, 0.02096210696049449, 0.05455522708949207, 0.0001343724805159903, 0.005778016662187585, 0.0012093523246439128, 0.0008062348830959419, 0.00040311744154797097, 0.008599838753023382, 0.0, 0.0001343724805159903, 0.0001343724805159903, 0.0005374899220639613, 0.0001343724805159903, 0.0006718624025799516, 0.0012093523246439128, 0.01195915076592314, 0.06342381080354743, 0.06019887127116366, 0.0001343724805159903, 0.007256113947863478, 0.0006718624025799515, 0.0009406073636119322, 0.0001343724805159903, 0.0067186240257995165, 0.011018543402311207, 0.013437248051599033, 0.0005374899220639613, 0.00819672131147541, 0.01195915076592314, 0.00819672131147541, 0.010077936038699274, 0.07739854877721042]
    # procesarePatchuri(path_imagine4, image4_file, image4_file1, image4_file2, '*.png', vect4retina, vect4exudat, image4_file3, lbp4retina, lbp4exudat, "retina", 4)
    # image4_file.close()
    # image4_file1.close()
    # image4_file2.close()
    # image4_file3.close()


def restrangereLBP(hist_LBP, nr_valori):

    LBP_restrans = []
    j = 0
    sum_valori = 0

    for i in range(len(hist_LBP)):
        if j < nr_valori:
            sum_valori += hist_LBP[i]
        else:
            LBP_restrans.append(sum_valori)
            j = 0
            sum_valori = round(hist_LBP[i], 3)
        j += 1

    sum_valori = 0
    i = 255
    while i >= 250:
        sum_valori += round(hist_LBP[i], 3)
        i -= 1

    LBP_restrans[9] += sum_valori

    return LBP_restrans


def concatenareImagini(path, image_number, latura_imagine):

    new_img = Image.new(mode = 'RGB', size = (4000, 4000))

    i = 0
    j = 0
    print("/------------------------------------------------")
    print("Padurea", image_number)
    for k in range(1, 401):
        image_path = path + str(k) + ".png"
        print("Patch-ul", image_path)
        img = Image.open(image_path)
        new_img.paste(img, (200 * i, 200 * j))
        i += 1
        if i == latura_imagine:
            j += 1
            i = 0
    print("Saving image..")
    print("-------------------------------------------------/")
    new_img_path = "paduri_concatenare/padure" + str(image_number) + ".png"
    new_img.save(new_img_path)

def concatenareImaginiRet(path, image_number, latura_imagine):

    new_img = Image.new(mode = 'RGB', size = (910, 910))

    i = 0
    j = 0
    print("/------------------------------------------------")
    print("Retina", image_number)
    k = 225
    while k > 0:
        image_path = path + str(k) + ".png"
        print("Patch-ul", image_path)
        img = Image.open(image_path)
        new_img.paste(img, (61 * i, 61 * j))
        i += 1
        if i == latura_imagine:
            j += 1
            i = 0
        k -= 1
    print("Saving image..")
    print("-------------------------------------------------/")
    new_img_path = "retina_procesare/retina" + str(image_number) + ".png"
    new_img.save(new_img_path)

def colorPatchuri(path, index_imagine):

    img = Image.open(path).convert('LA')
    file_name = "retina/split retina 4/" + str(index_imagine) + ".png"
    img.save(file_name)

def medie(arr):

    arrays = [np.array(x) for x in arr]
    lbpresult = [round(np.mean(k), 3) for k in zip(*arrays)]
    print(lbpresult)

def rotunjire(arr):

    for i in range(len(arr)):
        arr[i] = round(arr[i], 3)

    return arr


if __name__ == '__main__':

    #colorPatchuri("retina/split retina 4/185.png", 185)
    concatenareImaginiRet("retina/split retina 4/", 4, 15)

    lacun = {}
    Nr = []
    r = []
    lbparr_R = []
    lbparrcamp_R = []
    lbparr_G = []
    lbparrcamp_G = []
    lbparr_B = []
    lbparrcamp_B = []

    # start_time = time.time()
    # procesarePaduri()
    # print("--- %s seconde pentru procesarea imaginilor cu paduri ---" % (time.time() - start_time))

    #concatenareImagini("paduri_procesare/padure1/padure", 1, 20)
    # concatenareImagini("paduri_procesare/padure2/padure", 2, 20)
    # concatenareImagini("paduri_procesare/padure3/padure", 3, 20)
    # concatenareImagini("paduri_procesare/padure4/padure", 4, 20)
    # concatenareImagini("paduri_procesare/padure5/padure", 5, 20)


    # concatenareImaginiRet("retina/split retina 2/", 2, 15)
    # concatenareImaginiRet("retina/split retina 3/", 3, 15)
    # concatenareImaginiRet("retina/split retina 4/", 4, 15)
    # path = "retina\split retina 1"
    # img23 = path + "\\23.png"
    # print(img23)
    # lista_patchuri = []
    # colorPatchuri('retina\split retina 1', img23)

    #start_time = time.time()
    #procesareRetina()
    #print("--- %s seconde pentru procesarea imaginilor cu retina ---" % (time.time() - start_time))

    # arrays = [np.array(x) for x in lbparr_R]
    # lbpresult = [np.mean(k) for k in zip(*arrays)]
    # print("medie histograma R lbp clasa 1: ")
    # print(lbpresult)
    #
    # arrays1 = [np.array(x1) for x1 in lbparrcamp_R]
    # lbpresult1 = [np.mean(k1) for k1 in zip(*arrays1)]
    # print("medie histograma R lbp clasa 2: ")
    # print(lbpresult1)
    #
    # arrays2 = [np.array(x2) for x2 in lbparr_G]
    # lbpresult2 = [np.mean(k2) for k2 in zip(*arrays2)]
    # print("medie histograma G lbp clasa 1: ")
    # print(lbpresult2)
    #
    # arrays3 = [np.array(x3) for x3 in lbparrcamp_G]
    # lbpresult3 = [np.mean(k3) for k3 in zip(*arrays3)]
    # print("medie histograma G lbp clasa 2: ")
    # print(lbpresult3)
    #
    # arrays4 = [np.array(x4) for x4 in lbparr_B]
    # lbpresult4 = [np.mean(k4) for k4 in zip(*arrays4)]
    # print("medie histograma B lbp clasa 1: ")
    # print(lbpresult4)
    #
    # arrays5 = [np.array(x5) for x5 in lbparrcamp_B]
    # lbpresult5 = [np.mean(k5) for k5 in zip(*arrays5)]
    # print("medie histograma B lbp clasa 2: ")
    # print(lbpresult5)


