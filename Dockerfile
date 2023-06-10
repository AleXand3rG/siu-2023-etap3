FROM dudekw/siu-20.04

WORKDIR /root
COPY . /root/siu
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python ./get-pip.py 
RUN rm -f ./get-pip.pyd

#Install TensorFlow	
#RUN pip install --no-cache-dir TensorFlow	
#RUN apt-get install TensorFlow
COPY TurtleBG2.png /roads.png

