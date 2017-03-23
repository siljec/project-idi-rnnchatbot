import fasttext
import time
from preprocess_helpers import path_exists, merge_files, get_time


def create_fast_text_model(folder, merged_spellcheck_path):
    start_time_fasttext = time.time()
    model = fasttext.skipgram(merged_spellcheck_path, './'+folder+'/model')
    print("Time used to create Fasttext model: ", get_time(start_time_fasttext))
    return model


def preprocessing2(x_path, y_path, fast_text_training_file_path, force_create_new_files, folder="datafiles",force_train_fast_model_all_over = False):

    # -------------------------- Step 1: Merge file for fast text --------------------------
    if not path_exists(fast_text_training_file_path) or not force_create_new_files:
        merge_files(x_path=x_path, y_path=y_path, final_file=fast_text_training_file_path)


    # -------------------------- Step 2: Create fast text model --------------------------

    # If model exists, just read parameters in stead of training all over
    if path_exists("./"+folder+"/model.bin") and not force_train_fast_model_all_over:
        print("Load existing FastText model...")
        model = fasttext.load_model('./'+folder+'/model.bin', encoding='utf-8')
    else:
        print("Create FastText model...")
        model = create_fast_text_model(folder, merged_spellcheck_path=fast_text_training_file_path)

    return model
