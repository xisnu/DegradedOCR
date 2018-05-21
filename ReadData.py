from __future__ import print_function
import h5py,math
import numpy as np
import os,sys
from PIL import Image

#First
def readMainH5(h5file,maxw,maxh,readamount,write_sequences=False):
    #Reads a standard format h5 file and collects features, targets and sequence_lengths
    #maxw and maxh are maximum width(cols) and maximum height(rows) of all samples train+test+val
    f=h5py.File(h5file)
    samplenames=list(f.keys())
    total=len(samplenames)
    all_x=[]
    all_y=[]
    all_sampleid=[]
    seq_lengths=[]
    if(write_sequences):
        seq_file = open("Sequence_lengths", "w")
    for t in range(total):
            completed=(t/float(total))*100
            if (completed >= readamount):
                break
            sample=f.get(samplenames[t])
            target = sample.attrs["Reorder_Target"]
            sampleid=sample.attrs["SampleID"]
            sys.stdout.write("\rRead %s target %s-------Completed %0.2f " % (sampleid, target, completed))
            sys.stdout.flush()
            #testing correctness of target
            temp_target=target.split()
            target_length=len(temp_target)
            #if(target_length>=2):
            all_sampleid.append(sampleid)
            features=np.asarray(sample.get("Image")) # H x W
            seq = len(features[0])
            #Test if feature sequence is sufficiently long CTC requirement
            #if((seq/8.0)>(target_length*2)):
            #Now Resize Image to given Max_W Max_H
            padded_feature=pad_x_single(features,[maxh,maxw])
            all_x.append(padded_feature) # N x W x H x 1
            all_y.append(target)
            seq_lengths.append(seq)
            if(write_sequences):
                seq_file.write(sample.attrs["SampleID"]+","+str(seq) + "\n")


    print("Reading ",h5file," complete")
    if(write_sequences):
        seq_file.close()
    return all_x,all_y,seq_lengths,all_sampleid

#Second
def findDistinctCharacters(targets):
    '''
    Reads all targets (targets) and splits them to extract individual characters
    Creates an array of character-integer map (char_int)
    Finds the maximum target length
    Finds number of distinct characters (nbclasses)
    :param targets:
    :return char_int,max_target_length,nbclasses:
    '''
    total=len(targets)
    max_target_length=0
    char_int=[]
    all_chars=[]
    total_transcription_length=0 #Total number of characters
    for t in range(total):
        this_target=targets[t]
        chars=this_target.split()
        target_length=len(chars)
        total_transcription_length=total_transcription_length+target_length
        if(target_length>max_target_length):
            max_target_length=target_length
        for ch in chars:
            all_chars.append(ch)

    charset = list(set(all_chars))
    '''
    char_int.append("PD") #A special character representing padded value
    for ch in charset:
        char_int.append(ch)
    '''
    nbclasses = len(charset)
    print("Character Set processed for ", total, " data")
    print(charset)
    '''
    f=open("Character_Integer","w")
    for c in char_int:
        f.write(c+"\n")
    f.close()
    '''
    return charset, max_target_length, nbclasses, total_transcription_length

def pad_x_single(x,maxdim):
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([rows,cols,1])
    for r in range(len(x)):
        for c in range(len(x[r])):
            padded_x[r][c][0]=x[r][c]
    #print("\tPadding complete for ",total," data")
    return padded_x

#Third
def pad_x(x,maxdim):
    total=len(x)
    rows=maxdim[0]
    cols=maxdim[1]
    padded_x=np.zeros([total,rows,cols,1])
    for t in range(total):
        for r in range(len(x[t])):
            for c in range(len(x[t][r])):
                padded_x[t][r][c][0]=x[t][r][c]
    #print("\tPadding complete for ",total," data")
    return padded_x

#Call inside Training Module
def make_sparse_y(targets,char_int,max_target_length):
    total = len(targets)
    indices=[]
    values=[]
    shape=[total,max_target_length]
    for t in range(total):
        chars=targets[t].split()
        for c_pos in range(len(chars)):
            sparse_pos=[t,c_pos]
            sparse_val=char_int.index(chars[c_pos])
            indices.append(sparse_pos)
            values.append(sparse_val)
    return [indices,values,shape]


#Adjust Sequence lengths after CNN and Pooling
def adjustSequencelengths(seqlen,convstride,poolstride,maxtargetlength):
    total=len(seqlen)
    layers=len(convstride)
    for l in range(layers):
        for s in range(total):
            seqlen[s]=max(maxtargetlength,math.ceil(seqlen[s]/(convstride[l]*poolstride[l])))
    return seqlen

#Main
def load_data(trainh5,testh5,batchsize,generate_char_table):
    train_x, train_y, train_seq_lengths,train_sampleids=readMainH5(trainh5,2851,223,100,write_sequences=True)
    test_x,test_y,test_seq_lengths,test_sampleids=readMainH5(testh5,2851,223,100)

    sampleids=[train_sampleids,test_sampleids]

    train_charset, train_max_target_length, train_nbclasses,train_transcription_length=findDistinctCharacters(train_y)
    test_charset, test_max_target_length, test_nbclasses, test_trainscription_length = findDistinctCharacters(test_y)
    print("Train Char Set ", train_nbclasses, " Test Character set ", test_nbclasses)

    if (train_nbclasses < test_nbclasses):
        print("Warning ! Test set have more characters")

    train_charset.extend(test_charset)

    char_int = []
    if (generate_char_table):
        charset = list(set(train_charset))  # A combined Character set is created from Train and test Character set
        charset.sort()
        charset.insert(0, "PD")
        charset.append("BLANK")
        nb_classes = len(charset)  # For Blank

        for ch in charset:
            char_int.append(ch)

        ci = open("Character_Integer", "w")
        for ch in char_int:
            ci.write(ch + "\n")
        ci.close()
        print("Character Table Generated and Written")
    else:
        ci = open("Character_Integer")
        line = ci.readline()
        while line:
            char = line.strip("\n")
            char_int.append(char)
            line = ci.readline()
        nb_classes = len(char_int)
        print("Character Table Loaded from Generated File")
    print(char_int)

    max_target_length=max(train_max_target_length,test_max_target_length)
    max_seq_len=2851

    nbtrain=len(train_y)
    nbtest=len(train_y)

    y_train=[]
    y_test=[]

    batches=int(np.ceil(nbtrain/float(batchsize)))
    start=0
    for b in range(batches):
        end=min(nbtrain,start+batchsize)
        sparse_target=make_sparse_y(train_y[start:end],char_int,max_target_length)
        y_train.append(sparse_target)
        start=end

    batches = int(np.ceil(nbtest / float(batchsize)))
    start = 0
    for b in range(batches):
        end = min(nbtest, start + batchsize)
        sparse_target = make_sparse_y(test_y[start:end], char_int, max_target_length)
        y_test.append(sparse_target)
        start = end
    transcription_length=[train_transcription_length,test_trainscription_length]

    return [train_x,test_x],nb_classes,[train_seq_lengths,test_seq_lengths],[y_train,y_test],max_target_length,max_seq_len,char_int,transcription_length,sampleids

#Convert integer representation of string to unicode representation
def int_to_bangla(intarray,char_int_file,dbfile):
    '''
    Takes an array of integers (each representing a character as given in char_int_file
    dbfile contains global mapping
    :param intarray:
    :param char_int_file:
    :param dbfile:
    :return:unicode string,mapped character string
    '''
    char_int=[]
    f=open(char_int_file)
    line=f.readline()
    while line:
        info=line.strip("\n")
        char_int.append(info)
        line=f.readline()
    f.close()

    chars=[]
    #print("Intarray ",intarray)
    for i in intarray:
        chars.append(char_int[i])
    #print("Custom Classes ",chars)

    banglastring=""
    for ch in chars:
        f=open(dbfile)
        line=f.readline()
        while line:
            info=line.strip("\n").split(",")
            if(info[2]==ch):
                banglastring=banglastring+info[1]+" "
            line=f.readline()
        f.close()
    return banglastring,chars

def find_unicode_info(char,dbfile):
    #returns type and actual unicode position of a character
    f=open(dbfile)
    line=f.readline()
    type="v"
    pos="#"
    while line:
        info=line.strip("\n").split(",")
        #print(info)
        if(len(info)>5):
            #skip line
            line=f.readline()
        else:
            if(char==info[1]):#Found it in DB
                type=info[0]
                if(type=="m"):#its a modifier
                    pos=info[-1]
                break
            line=f.readline()
    f.close()
    return [type,pos]

def find_character_frequency(h5file,statfile):
    f=h5py.File(h5file)
    keys=list(f.keys())
    total=len(keys)
    targets=[]
    for t in range(total):
        sample=f.get(keys[t])
        target=sample.attrs["Custom_Target"]
        targets.append(target)
        print("Reading ",sample.attrs["SampleID"]," Target ",target)
    f.close()

    all_chars=[]
    for tg in targets:
        characters=tg.split()
        for ch in characters:
            all_chars.append(ch)

    unique_chars=list(set(all_chars))
    unique_chars.sort()
    nb_classes=len(unique_chars)
    hist=np.zeros([nb_classes])

    for tg in targets:
        characters=tg.split()
        for ch in characters:
            ind=unique_chars.index(ch)
            hist[ind]=hist[ind]+1

    print(unique_chars)
    print(hist)

    dict = open("/media/parthosarothi/OHWR/Dataset/Dict/bengalichardb.txt")
    line = dict.readline()
    unicode_custom=[]
    while line:
        info = line.strip("\n").split(",")
        if(len(info)==4) or (len(info)==5):
            unicode_map=[info[2],info[3]]
            unicode_custom.append(unicode_map)
        line=dict.readline()
    dict.close()

    outfile=open(statfile,"w")

    for i in range(nb_classes):
        for j in range(len(unicode_custom)):
            if(str(unique_chars[i])==unicode_custom[j][1]):
                unicode_value=str(unicode_custom[j][0])
                break
        outfile.write(unicode_value+","+str(unique_chars[i])+","+str(hist[i])+"\n")
    outfile.close()

def reset_unicode_order(unicodestring,dbfile):
    #Takes unicodestring seperated by space
    #returns properly ordered unicodestring
    unicodearray=unicodestring.split()
    unicodearray=[ch.decode("utf-8").encode("unicode-escape") for ch in unicodearray]
    nbchars=len(unicodearray)
    i=0
    while (i<nbchars-2):
        [type, pos]=find_unicode_info(unicodearray[i],dbfile)
        if(type=="m"):# May need swap
            if(pos=="p"):#swap
                temp=unicodearray[i]
                unicodearray[i]=unicodearray[i+1]
                unicodearray[i+1]=temp
                i=i+1
        i=i+1
    reorder_string=""
    for u in unicodearray:
        reorder_string=reorder_string+u.encode("utf-8").decode("unicode-escape")
    return reorder_string

def gather_offline_info(dir):
    widths=[]
    heights=[]
    for root,sd,files in os.walk(dir):
        for fname in files:
            print("Reading ",fname)
            if(fname[-4:]=="jpeg"):
                absfname=os.path.join(root,fname)
                im=Image.open(absfname)
                print(im.size) #cols, rows
                widths.append(im.size[0])
                heights.append(im.size[1])
    max_width=max(widths)
    max_height=max(heights)
    print("Max W=",max_width," Max H=",max_height)

'''
dbfile="/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/CompositeAndSingleCharacters.txt"
dbfile2="/media/parthosarothi/OHWR/Dataset/Dict/bengalichardb.txt"
char_int_file="/media/parthosarothi/OHWR/Experiments/BanglaRecognition/DegradedOffline/Character_Integer"
#gather_offline_info("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Test")
#find_character_frequency("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Train/train","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Train/trainstat")
test_intarray=[105,12,109,31,109,2,1,95,105,31,27,104,1,20,120,109,7,1
,25,27,1,105,2,115,15,104,1,105,12,37,104,1,122,15,115,1
,67,10,104,19,1,17,104,109,32,1,109,18,1,122,2,73,1,1
,1,0,0,0,0,0]
bs,_=int_to_bangla(test_intarray,char_int_file,dbfile)
print(bs)
us=reset_unicode_order(bs,dbfile2)
print(us)
'''