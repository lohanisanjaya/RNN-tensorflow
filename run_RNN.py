import main_RNN as RNN
import sys
local_rnn_lstm = RNN.Local_Rnn_LSTM(eta = 0.1, epochs = int(sys.argv[1]), \
      batch_size = 9,  
      time_steps = 10,
      num_inputs = 6, 
      hidden_units = 128, 
      num_classes = 6) #len(x_data)/num_inputs minus time-steps

local_rnn_lstm.Run_LSTM(data_file='data.pkl')
