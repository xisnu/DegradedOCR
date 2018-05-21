from __future__ import print_function
import os,shutil,csv,sys
sys.path.append("/media/parthosarothi/OHWR/Experiments/DatasetScripts/")
import h5py
from PIL import Image
import numpy as np
from UnicodeProcess import *
from random import shuffle

dbfile="Dict/bengalichardb.txt"
def split_train_test(dir,source,destination):
    all_index=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            print(fname)
            if(fname[-3:]=="txt"):
                index=fname.split("_")[1].split(".")[0]
                all_index.append(index)
    print("File list ready")
    total=0
    for root,sd,files in os.walk(source):
        for fname in files:
            if(fname[-3:]=="tif"):
                index = fname.split("_")[1].split(".")[0]
                for i in range(len(all_index)):
                    if(all_index[i]==index):
                        absfilename = os.path.join(root, fname)
                        shutil.move(absfilename, destination)
                        total = total + 1
                        break
                print("Moved ",fname)
    print('Total ',total,' files moved')

def group_image_with_groundtruth(dir):
    images=[]
    groundtruth=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            if(fname[-3:]=="tif"):
                images.append(os.path.join(root,fname))
            elif(fname[-3:]=="txt"):
                groundtruth.append(os.path.join(root,fname))
    print('File list ready')

    data=[]

    for gt in groundtruth:
        f=open(gt)
        line_no = 1
        line=f.readline()
        while line:
            target=line.strip("\n")
            fname=gt.split("/")[-1][:-4]
            image_name=fname+"_"+"line_"+str(line_no)+".tif"
            data.append([image_name,target])
            line_no+=1
            line=f.readline()
        f.close()

    outfile=open(dir+"/found.txt","w")
    nf=open(dir+"/notfound.txt","w")
    for d in data:
        imname=d[0]
        found=False
        for i in images:
            fname=i.split("/")[-1]
            if(fname==imname):
                outfile.write(i+","+d[1]+"\n")
                found=True
                break
        if(found==False):
            nf.write(imname+","+d[1]+"\n")
    outfile.close()
    nf.close()

def read_D2_format(dir):
    images = []
    groundtruth = []
    for root, sd, files in os.walk(dir):
        for fname in files:
            if (fname[-3:] == "csv"):
                groundtruth.append([root,os.path.join(root, fname)])
    print('File list ready')

    outfile=open(dir+"/Outfile","w")

    for gt in groundtruth:
        f = open(gt[1])
        reader=csv.reader(f,delimiter="@")
        for row in reader:
            try:
                print(row[0])
                imagefilename=gt[0]+"/"+row[0]
                groundtruth=row[1]
                outfile.write(str(imagefilename)+"@"+str(groundtruth)+"\n")
            except:
                pass
        f.close()
    outfile.close()
    print("outputfile Ready")

def unicode_to_hex(dbfile,char):
    f=open(dbfile)
    line=f.readline()
    custom=["pn",1]
    while line:
        info=line.strip("\n").split(",")
        if(info[1]==char):
            if(len(info)==8):
                custom=[info[5],info[6],2]
            else:
                custom=[info[3],1]
            break
        line=f.readline()
    return custom

def makeh5_from_dir(dir,outfile):
    groundtruth=dir+"/Groundtruth.txt"
    f=h5py.File(outfile,"w")
    datafile=open(groundtruth)
    line=datafile.readline()
    global dbfile
    while line:
        info=line.strip("\n").split("@")
        filename_rel= info[0].split("/")[-1]
        filename = dir + "/Line_Images/" + filename_rel
        if(os.path.isfile(filename)):
            #print("Found-",filename)
            im=Image.open(filename).convert("L")
            cols,rows=im.size
            pixels=np.reshape(im.getdata(),[rows,cols])
            group=f.create_group(filename_rel)
            group.create_dataset("Image",data=pixels)
            unicode_target=info[1]
            dbfile="Dict/AllCharcaters.txt"
            compounddbfile="Dict/BanglaCompositeMap.txt"
            unicode_line = convert_bangla_line_to_unicode(unicode_target)
            unicode_compound_map = replace_compound_in_unicode_line(unicode_line, compounddbfile)
            custom_line = convert_unicode_line_to_custom(unicode_compound_map, dbfile)
            reorderlist = ['m3', 'm8', 'm9']
            reorder_line = reorder_modifier_in_custom_line(custom_line, reorderlist)
            print("Reading ", filename," target ",unicode_target," Custsom ",reorder_line)
            group.attrs["Bangla_Target"] = unicode_target
            group.attrs["Unicode_Target"]=unicode_line
            group.attrs["Custom_Target"] = custom_line
            group.attrs["Reorder_Target"] = reorder_line
            group.attrs["SampleID"]=filename
            line=datafile.readline()
        else:
            line = datafile.readline()

    f.close()
    datafile.close()
    print("HDF file ready")

def find_character_histogram(charlist,h5file,statfile):
    #charlist contains all single and composite characters
    f=open(charlist)
    line=f.readline()
    map=[]
    bangla=[]
    charcount=0
    while line:
        info=line.strip("\n").split(",")
        print(info)
        map.append(info[-1])
        bangla.append(info[1])
        charcount+=1
        line=f.readline()
    f.close()
    print("All Character List Ready")
    hist=np.zeros([charcount])
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs["Reorder_Target"]
        words=target.split("*")
        for w in words:
            chars=w.split()
            for ch in chars:
                try:
                    ind=map.index(ch)
                    hist[ind]+=1
                except:
                    pass
    f.close()
    f=open(statfile,"w")
    for i in range(charcount):
        f.write(bangla[i]+","+map[i]+","+str(hist[i])+"\n")
    f.close()
    print("Complete")

def append_single_characters(outfile,chardir,charfile):
    f=open(outfile,"a")
    for root,sd,files in os.walk(chardir):
        for fname in files:
            filename=os.path.join(root,fname)
            label=filename.split("/")[-2]
            fc=open(charfile)
            line=fc.readline()
            while line:
                line=line.strip("\n").split(",")
                if(label==line[0]):
                    unicode_tag=line[-1]
                    break
                line=fc.readline()
            fc.close()
            print(filename,"--",label,"--",unicode_tag)
            f.write(filename+"@"+unicode_tag+"\n")
    f.close()
    print("Completed")

def shuffle_outfile(outfile):
    filelist=[]
    f=open(outfile)
    line=f.readline()
    while line:
        filelist.append(line)
        line=f.readline()
    f.close()

    shuffle(filelist)

    print("List Shuffled")

    f=open(outfile,"w")
    for l in filelist:
        f.write(l+"\n")
    f.close()
    print("Outfile Updated")

def find_distinct_words(dir):
    all_words=[]
    for root,sd,files in os.walk(dir):
        for filename in files:
            abs_fname=os.path.join(root,filename)
            ftype=abs_fname[-3:]
            if(ftype=="txt"):
                #now read the file
                f=open(abs_fname)
                line=f.readline()
                while line:
                    line=line.strip("\n")
                    words=line.split()
                    all_words.extend(words)
                    line=f.readline()
                f.close()
                print("File ",abs_fname," Reading complete")
    wordset=list(set(all_words))
    f=open("Wordset.txt","w")
    for w in wordset:
        f.write(str(w)+"\n")
    f.close()
    print("Wordset Ready")

def find_distinct_words_hdf(h5file):
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    all_words=[]
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs['Bangla_Target']
        words=target.split()
        for w in words:
            all_words.append(w.replace(" ",""))
        print('Reading ',sample.attrs["SampleID"])
    f.close()
    return all_words


def find_distinct_words_allhdf(hdfs):
    all_words=[]
    for fn in hdfs:
        all_w=find_distinct_words_hdf(fn)
        all_words.extend(all_w)
        print('Reading ',fn)

    wordset = list(set(all_words))
    f = open("wordseth5.txt", "w")
    for w in wordset:
        f.write(w + "\n")
    f.close()


dir="Data/Sample/"
source="/media/parthosarothi/OHWR/Dataset/ICBOCR-D1/All"

#split_train_test(dir,source,dir)
#group_image_with_groundtruth(dir)
#read_D2_format(dir)
makeh5_from_dir(dir,"Data/train")
#find_character_histogram(dir+"/CompositeAndSingleCharacters.txt",dir+"/Train/train",dir+"/Train/trainstat")
#append_single_characters("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Train/Outfile","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/SingleCharacters","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/BengaliCharSymbolListForLineRecognition.txt")
#shuffle_outfile(dir+"/Outfile")
#find_distinct_words("/media/parthosarothi/OHWR/Dataset/ICBOCR-D4")
#hdfs=["/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/train","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/test"]
#find_distinct_words_allhdf(hdfs)