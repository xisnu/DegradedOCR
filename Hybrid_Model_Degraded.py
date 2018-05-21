from __future__ import print_function
import tensorflow as tf
from ReadData import *
import time,sys

class CNNBLSTM:

    def __init__(self,max_timesteps,nb_features,nb_classes,modelname):
        self.hybridmodel=tf.Graph()
        self.max_timesteps=max_timesteps
        self.nb_features=nb_features
        self.nb_classes=nb_classes
        self.model_name=modelname
        print("Empty Graph Created")

    def get_layer_shape(self, layer):
        thisshape = tf.Tensor.get_shape(layer)
        ts = [thisshape[i].value for i in range(len(thisshape))]
        return ts

    def readNetworkStructure(self,configfile):
        nw = {}
        f = open(configfile)
        line = f.readline()
        while line:
            info = line.strip("\n").split(",")
            nw[info[0]] = info[1]
            line = f.readline()
        self.filterwidth=int(nw['filterwidth'])
        self.filterheight = int(nw['filterheight'])
        self.nb_filters=[int(fi) for fi in nw['nb_filters'].split()]
        self.conv_stride = [int(fi) for fi in nw['conv_stride'].split()]
        self.pool_stride = [int(fi) for fi in nw['pool_stride'].split()]
        self.nb_hidden=int(nw['nb_hidden'])
        self.lr=float(nw['lr'])
        print('Network Configuration Understood')

    def createNetwork(self,configfile):
        self.readNetworkStructure(configfile)
        with self.hybridmodel.as_default():
            # tf.reset_default_graph()
            self.network_input_x = tf.placeholder(tf.float32, [None, self.filterheight, self.max_timesteps, self.nb_features])
            self.network_target_y = tf.sparse_placeholder(tf.int32)
            self.network_input_sequence_length = tf.placeholder(tf.int32, [None])
            #seq_len2 = tf.placeholder(tf.int32, [None])

            f1 = [self.filterheight, self.filterwidth, self.nb_features, self.nb_filters[0]]
            W1 = tf.Variable(tf.truncated_normal(f1, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[0]]), name="b1")
            conv_1 = tf.nn.conv2d(self.network_input_x, W1, strides=[1, self.filterheight, self.conv_stride[0], 1], padding='SAME')
            nl_1 = tf.nn.relu(tf.add(conv_1, b1))  # batchsize,1,maxsteps,filter1

            f1_double= [self.filterheight, self.filterwidth*2, self.nb_features, self.nb_filters[1]]
            W1_double = tf.Variable(tf.truncated_normal(f1_double, stddev=0.1), name="W1_double")
            b1_double = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[1]]), name="b1_double")
            conv_1_double = tf.nn.conv2d(self.network_input_x, W1_double, strides=[1, self.filterheight, self.conv_stride[0], 1],padding='SAME')
            nl_1_double = tf.nn.relu(tf.add(conv_1_double, b1_double))

            f1_triple = [self.filterheight, self.filterwidth * 3, self.nb_features, self.nb_filters[2]]
            W1_triple = tf.Variable(tf.truncated_normal(f1_triple, stddev=0.1), name="W1_triple")
            b1_triple = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[2]]), name="b1_triple")
            conv_1_triple = tf.nn.conv2d(self.network_input_x, W1_triple,strides=[1, self.filterheight, self.conv_stride[0], 1], padding='SAME')
            nl_1_triple = tf.nn.relu(tf.add(conv_1_triple, b1_triple))

            merge_conv1=tf.concat([nl_1,nl_1_double,nl_1_triple],3)
            shape = self.get_layer_shape(merge_conv1)
            print("First Inception Block = ", shape)


            mp_1 = tf.nn.max_pool(nl_1, ksize=[1, 1, 7, 1], strides=[1, 1, self.pool_stride[0], 1],padding='SAME')  # batchsize,1,maxsteps,filter1
            shape=self.get_layer_shape(mp_1)
            print("First Conv Block = ",shape)
            # ---------------1st Conv MP Block Ends----------------------#

            # ---------------2nd Block Starts---------------------#
            f2 = [1, self.filterwidth, self.nb_filters[0], self.nb_filters[1]]
            W2 = tf.Variable(tf.truncated_normal(f2, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[1]]), name="b2")
            conv_2 = tf.nn.conv2d(mp_1, W2, strides=[1, 1, self.conv_stride[1], 1], padding='SAME')
            nl_2 = tf.nn.relu(tf.add(conv_2, b2))  # batchsize,1,maxsteps,filter2
            mp_2 = tf.nn.max_pool(nl_2, ksize=[1, 1, 5, 1], strides=[1, 1, self.pool_stride[1], 1],padding='SAME')  # batchsize,1,maxsteps,filter2
            shape = self.get_layer_shape(mp_2)
            print("Second Conv Block = ", shape)
            # --------------2nd Conv MP Block Ends-----------------------#


            # ---------------3rd Block Starts---------------------#
            f3 = [1, self.filterwidth, self.nb_filters[1], self.nb_filters[2]]
            W3 = tf.Variable(tf.truncated_normal(f3, stddev=0.1), name="W3")
            b3 = tf.Variable(tf.constant(0.1, shape=[self.nb_filters[2]]), name="b3")
            conv_3 = tf.nn.conv2d(mp_2, W3, strides=[1, 1, self.conv_stride[2], 1], padding='SAME')
            nl_3 = tf.nn.relu(tf.add(conv_3, b3))  # batchsize,1,maxsteps,filter2
            mp_3 = tf.nn.max_pool(nl_3, ksize=[1, 1, 5, 1], strides=[1, 1, self.pool_stride[2], 1],padding='SAME')  # batchsize,1,maxsteps,filter3
            shape = self.get_layer_shape(mp_3)
            print("Third Conv Block = ", shape)
            # --------------3rd Conv MP Block Ends-----------------------#


            conv_reshape = tf.squeeze(mp_3, squeeze_dims=[1])  # batchsize,maxsteps,filter3
            shape = self.get_layer_shape(conv_reshape)
            print("CNN --> RNN Reshape = ", shape)

            with tf.variable_scope("cell_def_1"):
                f_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)
                b_cell = tf.nn.rnn_cell.LSTMCell(self.nb_hidden, state_is_tuple=True)

            with tf.variable_scope("cell_op_1"):
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, conv_reshape, sequence_length=self.network_input_sequence_length,dtype=tf.float32)

            merge = tf.concat(outputs,2)
            shape = self.get_layer_shape(merge)
            print("First BLSTM = ", shape)

            nb_hidden_2=self.nb_hidden*2

            with tf.variable_scope("cell_def_2"):
                f1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)
                b1_cell = tf.nn.rnn_cell.LSTMCell(nb_hidden_2, state_is_tuple=True)

            with tf.variable_scope("cell_op_2"):
                outputs2, _ = tf.nn.bidirectional_dynamic_rnn(f1_cell, b1_cell, merge, sequence_length=self.network_input_sequence_length, dtype=tf.float32)

            merge2 = tf.concat(outputs2, 2)
            shape = self.get_layer_shape(merge2)
            print("Second BLSTM = ", shape)
            batch_s, timesteps = shape[0], shape[1]
            print(timesteps)

            blstm_features=shape[-1]


            output_reshape = tf.reshape(merge2, [-1, blstm_features])  # maxsteps*batchsize,nb_hidden
            shape = self.get_layer_shape(output_reshape)
            print("RNN Time Squeezed = ", shape)

            W = tf.Variable(tf.truncated_normal([blstm_features, self.nb_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0., shape=[self.nb_classes]), name="b")

            logits = tf.matmul(output_reshape, W) + b  # maxsteps*batchsize,nb_classes
            logits=tf.reshape(logits, [-1, timesteps, self.nb_classes])
            shape = self.get_layer_shape(logits)
            print("Logits = ", shape)

            logits_reshape = tf.transpose(logits,[1, 0, 2])  # maxsteps,batchsize,nb_classes
            shape = self.get_layer_shape(logits_reshape)
            print("RNN Time Distributed (Time Major) = ", shape)

            loss = tf.nn.ctc_loss(self.network_target_y, logits_reshape, self.network_input_sequence_length)
            self.cost = tf.reduce_mean(loss)

            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.cost)

            # for greedy decoder input(i.e. logits_reshape) must be of shape maxtime,batchsize,nb_classes
            # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits_reshape, seq_len)--very slow
            decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_reshape, self.network_input_sequence_length)

            self.decoded_words=tf.sparse_to_dense(decoded[0].indices,decoded[0].dense_shape,decoded[0].values)
            self.actual_targets=tf.sparse_to_dense(self.network_target_y.indices,self.network_target_y.dense_shape,self.network_target_y.values)

            actual_ed = tf.edit_distance(tf.cast(decoded[0], tf.int32), self.network_target_y, normalize=False)
            self.ler = tf.reduce_sum(actual_ed) #insertion+deletion+substitution
            self.new_saver = tf.train.Saver()
            print("Network Ready")

    def trainNetwork(self,nb_epochs,batchsize,x,y,seqlen,max_target_length,transcription_length,weightfiles,mode):
        x_train=x[0]
        x_test=x[1]
        y_train=y[0]
        y_test=y[1]
        seq_len_train=adjustSequencelengths(seqlen[0],self.conv_stride,self.pool_stride,max_target_length)
        seq_len_test = adjustSequencelengths(seqlen[1],self.conv_stride,self.pool_stride,max_target_length)
        weightfile_last=weightfiles[0]
        weightfile_best = weightfiles[1]
        train_transcription_length=transcription_length[0]
        test_transcription_length=transcription_length[1]

        with tf.Session(graph=self.hybridmodel) as session:
            if(mode=="New"):
                init_op = tf.global_variables_initializer()
                session.run(init_op)
                print("New Weights Initiated")
            elif(mode=="Load"):
                self.new_saver.restore(session, weightfile_best)
                print("Previous weights loaded")
            else:
                print("Unknown Mode")
                return
            nb_train=len(x_train)
            nb_test=len(x_test)
            trainbatch=int(np.ceil(float(nb_train)/batchsize))
            testbatch=int(np.ceil(float(nb_test)/batchsize))
            besttestacc=0
            for e in range(nb_epochs):
                totalloss = 0
                totalacc = 0
                starttime = time.time()
                train_batch_start=0
                logf = open("Training_log", "a")
                for b in range(trainbatch):
                    train_batch_end=min(nb_train,train_batch_start+batchsize)
                    sys.stdout.write("\rTraining Batch %d / %d" %(b,trainbatch))
                    sys.stdout.flush()
                    batch_x=np.asarray(x_train[train_batch_start:train_batch_end])
                    batch_seq_len=seq_len_train[train_batch_start:train_batch_end]
                    batch_target_sparse=y_train[b]

                    feed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse, self.network_input_sequence_length: batch_seq_len}

                    batchloss, batchacc, _ = session.run([self.cost, self.ler, self.optimizer], feed)

                    totalloss = totalloss + batchloss
                    totalacc = totalacc + batchacc
                    train_batch_start=train_batch_end

                trainloss = totalloss / trainbatch
                #avgacc = totalacc / trainbatch
                print("\nTraining Edit Distance ",totalacc,"/",train_transcription_length)
                trainacc=(1-(float(totalacc)/train_transcription_length))*100
                # Now save the model
                self.new_saver.save(session, weightfile_last)

                testloss = 0
                testacc = 0

                test_batch_start = 0
                output_words=[]
                target_words=[]
                for b in range(testbatch):
                    test_batch_end = min(nb_test, test_batch_start + batchsize)
                    sys.stdout.write("\rTesting Batch %d/%d"%(b,testbatch) )
                    sys.stdout.flush()
                    batch_x = np.asarray(x_test[test_batch_start:test_batch_end])
                    batch_seq_len = seq_len_test[test_batch_start:test_batch_end]
                    batch_target_sparse = y_test[b]

                    testfeed = {self.network_input_x: batch_x, self.network_target_y: batch_target_sparse,self.network_input_sequence_length: batch_seq_len}

                    try:
                        batchloss, batchacc,output_words_batch,target_words_batch = session.run([self.cost, self.ler,self.decoded_words,self.actual_targets], testfeed)
                    except:
                        pass
                    output_words.extend(output_words_batch)
                    target_words.extend(target_words_batch)
                    testloss = testloss + batchloss
                    testacc = testacc + batchacc
                    test_batch_start=test_batch_end

                testloss = testloss / testbatch
                testacc=(1-(float(testacc)/test_transcription_length))*100

                result=open("Decoded","w")
                corrects=0.0
                for w in range(nb_test):
                    result.write(str(target_words[w])+","+str(output_words[w])+"\n")
                    if(len(output_words[w])>=len(target_words[w])):
                        flag=False
                        for c in range(len(target_words[w])):
                            if(output_words[w][c]==target_words[w][c]):
                                flag=True
                            else:
                                flag=False
                                break
                        if(flag==True):
                            corrects=corrects+1
                result.close()

                avg_word_accuracy=(corrects/nb_test)*100

                if (testacc > besttestacc):
                    besttestacc = testacc
                    print("\nNetwork Improvement")
                    self.new_saver.save(session, weightfile_best)
                endtime = time.time()
                timetaken = endtime - starttime
                msg = "\nEpoch " + str(e) + "(" + str(timetaken) + " sec) Training: Loss is " + str(trainloss) + " Accuracy " + str(trainacc) + "% Testing: Loss " + str(testloss) + " Accuracy " + str(testacc) + "% Best " + str(besttestacc) + "%\n"
                print(msg)
                logf.write(msg)
                logf.write("\nWord Accuracy"+str(avg_word_accuracy))
                logf.close()
                msg="Word Accuracy="+str(avg_word_accuracy)
                print(msg)

    def predict(self,input_data,sequence_length,weightfile,dbfile,max_target_length,sampleids):
        #print(input_data[1])
        target_y=input_data[1][0]
        nb_predicts = target_y[2][0]
        print("Number of test cases ",nb_predicts)
        sequence_length = sequence_length[:nb_predicts]
        sequence_length = adjustSequencelengths(sequence_length, self.conv_stride, self.pool_stride, max_target_length)
        sequenceids=sampleids[:nb_predicts]
        print("input_x=", len(input_data[0]), " Sequence Length=", len(sequence_length), " Target ", len(target_y[0]),len(target_y[1]),target_y[2])
        with tf.Session(graph=self.hybridmodel) as predict_session:
            self.new_saver.restore(predict_session,weightfile)
            print("Saved Model Loaded")
            feed={self.network_input_x:input_data[0][:nb_predicts],self.network_input_sequence_length:sequence_length,self.network_target_y:target_y}
            actual_words,output_words=predict_session.run([self.actual_targets,self.decoded_words],feed)
            predicted_words=[]
            actual_targets=[]
            total=len(output_words)
            for w in range(total):
                word=output_words[w]
                unicode_output, _ = int_to_bangla(word, "Character_Integer",dbfile)
                unicode_output = reset_unicode_order(unicode_output,charposfile)
                predicted_words.append(unicode_output)

                word=actual_words[w]
                unicode_output, _ = int_to_bangla(word, "Character_Integer", dbfile)
                unicode_output = reset_unicode_order(unicode_output, charposfile)
                actual_targets.append(unicode_output)
            f=open("Predicted.txt","w")
            for w in range(len(predicted_words)) :
                f.write(sequenceids[w]+","+actual_targets[w].encode('utf-8')+","+predicted_words[w].encode('utf-8')+"\n")
            f.close()
            print("Output Ready")
            return [predicted_words]

'''
Model should be fed with
Max_Time_steps,nb_features,nb_classes
Rest of the parameters written in Config file
'''
def main(task,mode,dbfile,files,weightfile,batchsize):
    generate_char_table = True
    if (mode == "Load"):
        generate_char_table = False
    [x_train, x_test], nb_classes, seqlen, [train_y,test_y], max_target_length, max_seq_length, char_int, transcription_length,sampleids = load_data(
        files[0], files[1], batchsize, generate_char_table)
    nb_features = 1
    x = [x_train, x_test]
    y = [train_y, test_y]

    print("Training Data X=", len(x_train), " Testing Data X=", len(x_test))
    print(" Max Seq len=", max_seq_length, " Max Target length=", max_target_length)
    print("Number of classes (including blank)", nb_classes)

    model = CNNBLSTM(max_seq_length, nb_features, nb_classes, "Hybrid")
    model.createNetwork("Config")

    weightfile_last = weightfile + "/last"
    weightfile_best = weightfile + "/Best/best"
    weightfiles = [weightfile_last, weightfile_best]
    if(task=="Train"):
        #train network
        model.trainNetwork(100, batchsize, x, y, seqlen, max_target_length, transcription_length, weightfiles, mode)
    elif(task=="Predict"):
        #test network
        x_test=np.asarray(x_test)
        #print("XTest = ",x_test.shape," YTest=",len(test_y))
        model.predict([x_test,test_y], seqlen[1], weightfile_best, dbfile, max_target_length,sampleids[1])


dbfile = "Dict/CompositeAndSingleCharacters.txt"
charposfile="Dict//bengalichardb.txt"
files = ["Data/train","Data/train"]
weightfile = "Weights"
batchsize = 16

task="Train"
mode="New"

main(task,mode,dbfile,files,weightfile,batchsize)