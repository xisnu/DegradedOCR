from __future__ import print_function
import csv
#True target character (TTC) example \u09xx
#Minimal target doesnor have \u09
def getcustomfromtrue(ch,dbfile):#input TTC and get custom char
    f=open(dbfile,'r')
    reader=csv.reader(f)
    cust="NA"
    for row in reader:
        if(row[1]==ch):
            cust=row[3]
            break
    return cust

def getttcfromminimal(target):#input Minimal label and get TTC label
    truetarget=""
    targetlen=len(target)
    t=0
    while(t<targetlen-1):
        truetarget=truetarget+"\\u09"+target[t]+target[t+1]+" "
        t=t+2
    return truetarget.rstrip()

def getunicodefromtrue(ch,dbfile):#input TTC and get unicoded char
    f=open(dbfile,'r')
    reader=csv.reader(f)
    uni="NA"
    for row in reader:
        if(row[1]==ch):
            uni=row[2]
            break
    return uni

def labeltruetocustom(label,dbfile):#input true target lebel (sequence of TTC seperated by space) and get custom label (seperated by space)
    chars=label.split(" ")
    #print "To find ",label," Chars ",len(chars)
    target=""
    for c in chars:
        op=getcustomfromtrue(c,dbfile)
        if(op!="NA"):
            target=target+op+" "
    return target.rstrip()

def labeltruetounicode(label,dbfile):#input true target lebel (sequence of TTC seperated by space) and get Unicode label
    chars=label.split(" ")
    #print "To find ",label," Chars ",len(chars)
    target=""
    for c in chars:
        op=getunicodefromtrue(c,dbfile)
        if(op!="NA"):
            target=target+op
    return target

def process_compound_characters(filename,outfile):
    #filename may contain distinct composite characters
    all_chars=[]
    f=open(filename)
    line=f.readline()
    while line:
        char=line.strip("\n")
        unicoded=char.decode("utf-8").encode("unicode-escape")
        unicode_characters=unicoded.split("\\")[1:]
        unicode_space_separated_string=""
        for ch in unicode_characters:
               unicode_space_separated_string=unicode_space_separated_string+"\\"+ch+" "
        all_chars.append(unicode_space_separated_string)
        line = f.readline()

    charset=list(set(all_chars))
    print("Distinct Characters Ready")

    wf = open(outfile, "w")
    for us in charset:
        chars=us.split()
        bangla=""
        strlen=0
        for ch in chars:
            bangla_ch=ch.encode("utf-8").decode("unicode-escape")
            bangla=bangla+bangla_ch
            strlen+=1
        #print("Compound Character ",bangla,unicode_characters,unicode_space_separated_string)
        wf.write(bangla.encode("utf-8")+","+us+","+str(strlen)+"\n")
    print("Final Compound Character List is Ready")

    wf.close()
    f.close()

def map_compound_to_custom(compoundfile,mapfile):
    f=open(compoundfile)
    wf=open(mapfile,"w")
    line=f.readline()
    index=1
    while line:
        info=line.strip("\n")
        wf.write(info+","+"cmp"+str(index)+"\n")
        index+=1
        line=f.readline()
    f.close()
    wf.close()

def find_distinct_chars_in_corpus(corpusfile,charset):
    f=open(corpusfile)
    wf=open(charset,"w")
    all_chars=[]
    line=f.readline()
    while line:
        info=line.strip("\n")
        words=info.split()
        for w in words:
            unicoded=w.decode("utf-8").encode("unicode-escape")
            unicode_characters=unicoded.split("\\")[1:]
            for uc in unicode_characters:
                uc=uc[:5]
                all_chars.append("\\"+uc)
        print("Reading ",info)
        line=f.readline()
    charset=list(set(all_chars))
    print(charset)
    print("Total=",len(charset))
    for ch in charset:
        bangla_char=""+ch.encode("utf-8").decode("unicode-escape")
        wf.write(ch+","+bangla_char.encode("utf-8")+"\n")
    wf.close()


def convert_corpus_to_unicode(corpusfile,unicodecorpusfile):
    f=open(corpusfile)
    wf=open(unicodecorpusfile,"w")
    line=f.readline()
    while line:
        #find words in line
        words=line.strip("\n").split()
        #find char in words
        for w in words:
            unicoded = w.decode("utf-8").encode("unicode-escape")
            unicode_characters = unicoded.split("\\")[1:]
            for uc in unicode_characters:
                uc = uc[:5]
                wf.write("\\"+uc+" ")
            wf.write("* ")#Character separator in word
        wf.write("\n")#Word separator in line
        line=f.readline()
    f.close()
    wf.close()

def convert_bangla_line_to_unicode(line):
    #unicode line words are seperated by * characters are seperated by space
    words=line.split()
    unicode_line=""
    for w in words:
        unicoded = w.decode("utf-8").encode("unicode-escape")
        unicode_characters = unicoded.split("\\")[1:]
        for uc in unicode_characters:
            uc = uc[:5]
            unicode_line=unicode_line+"\\"+uc+" "
        unicode_line=unicode_line+"* "
    return unicode_line

def convert_unicode_line_to_custom(line,dbfile):
    #custom line words are seperated by * characters are separated by space
    words=line.split("*")
    custom_line=""
    for w in words:
        #print("\t\tReading word ",w)
        characters=w.split()
        for ch in characters:
            custom_tag=ch
            f = open(dbfile)
            line=f.readline()
            # Looking for a custom tag for this unicode character from dbfile
            #if not found then original character will be retained
            while line:
                info=line.strip("\n").split(",")
                if(ch==info[0]):
                    custom_tag=info[-1]
                    break
                else:
                    line=f.readline()
            f.close()
            custom_line = custom_line + custom_tag + " "
            #print("\t\t\tReading character ", ch," tag ",custom_tag)
        custom_line=custom_line+"* "
    return custom_line

def replace_compound_in_unicode_line(line,compounddbfile):
    f=open(compounddbfile)
    ln=f.readline()
    while ln:
        info=ln.strip("\n").split(",")
        compound_string=info[1].rstrip()
        #print("Compound String ",compound_string)
        compound_tag=info[-1]
        line=line.replace(compound_string,compound_tag)
        ln=f.readline()
    return line

def reorder_modifier_in_custom_line(line,reorderlist):
    #reorder list contains custom label of those modifier that has different phonetic positions
    #line has custom labels
    reorder_line=""
    words=line.split("*")
    for w in words:
        #find characters in words
        chars=w.split()
        reposition=False
        for c in range(len(chars)):
            for m in reorderlist:
                if(m==chars[c]):
                    reposition=True
                    break
            if(reposition):
                temp=chars[c-1]
                chars[c-1]=chars[c]
                chars[c]=temp
                reposition=False
        for ch in chars:
            reorder_line+=ch+" "
        reorder_line+="* "
    return reorder_line


def map_corpus_to_custom(corpusfile,dbfile,compounddbfile):
    f=open(corpusfile)
    line=f.readline()
    while line:
        line=line.strip("\n")
        print("Reading ",line)
        unicode_line=convert_bangla_line_to_unicode(line)
        unicode_compound_map=replace_compound_in_unicode_line(unicode_line,compounddbfile)
        custom_line=convert_unicode_line_to_custom(unicode_compound_map,dbfile)
        reorderlist=['m3','m8','m9']
        reorder_line=reorder_modifier_in_custom_line(custom_line,reorderlist)
        print("\tUnicoded Line=",unicode_line)
        print("\tCompound mapped=",unicode_compound_map)
        print("\tCustom Line=",custom_line)
        print("\tReorder Line=", reorder_line)
        line=f.readline()
    f.close()

#process_compound_characters("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/CompositeCharacters.txt","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Composite_Charset.txt")
#map_compound_to_custom("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/Composite_Charset.txt","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/BanglaCompositeMap.txt")
#find_distinct_chars_in_corpus("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/AllBook.txt","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/AllCharcaters.txt")
#convert_corpus_to_unicode("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/AllBook.txt","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/UnicodeCorpus.txt")
#map_corpus_to_custom("/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/testcorpus","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/AllCharcaters.txt","/media/parthosarothi/OHWR/Dataset/ICBOCR-D2/BanglaCompositeMap.txt")
