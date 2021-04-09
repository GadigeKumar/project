
# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
!pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime
from google.colab import files

!nvidia-smi



gpt2.download_gpt2(model_name="124M")



gpt2.mount_gdrive()



file_name = "alllines.txt"



gpt2.copy_file_from_gdrive(file_name)


sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='124M',
              steps=1000,
              restore_from='fresh',
              run_name='run1',
              print_every=10,
              sample_every=200,
              save_every=500
              )


gpt2.copy_checkpoint_to_gdrive(run_name='run1')



gpt2.copy_checkpoint_from_gdrive(run_name='run1')


sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name='run1')



gpt2.generate(sess, run_name='run1')



gpt2.generate(sess,
              length=5,
              temperature=0.7,
              prefix="Thank you",
              nsamples=5,
              batch_size=5
              )


gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=500,
                      temperature=0.7,
                      nsamples=100,
                      batch_size=20
                      )

# may have to run twice to get file to download
files.download(gen_file)


model_name = "774M"

gpt2.download_gpt2(model_name=model_name)

sess = gpt2.start_tf_sess()

gpt2.load_gpt2(sess, model_name=model_name)

gpt2.generate(sess,
              model_name=model_name,
              prefix="The secret of life is",
              length=100,
              temperature=0.7,
              top_p=0.9,
              nsamples=5,
              batch_size=5
              )


