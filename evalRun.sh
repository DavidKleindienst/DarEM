#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate tfOD

python  model_main_tf2.py --pipeline_config_path=my_models/efficientnet_d0/pipeline.config \ 
         --model_dir=my_models/efficientnet_d0/split1024_noObjLongTraining \
         --checkpoint_dir=my_models/efficientnet_d0/split1024_noObjLongTraining/

python  model_main_tf2.py --pipeline_config_path=my_models/efficientnet_d1/pipeline.config \ 
         --model_dir=my_models/efficientnet_d1/split1024_noObj \
         --checkpoint_dir=my_models/efficientnet_d1/split1024_noObj/

python  model_main_tf2.py --pipeline_config_path=my_models/efficientnet_d2/pipeline.config \ 
         --model_dir=my_models/efficientnet_d2/split1024_noObj \
         --checkpoint_dir=my_models/efficientnet_d2/split1024_noObj/


python  model_main_tf2.py --pipeline_config_path=my_models/efficientnet_d3/pipeline.config \
	 --model_dir=my_models/efficientnet_d3/split1024_noObj \
         --checkpoint_dir=my_models/efficientnet_d3/split1024_noObj/

