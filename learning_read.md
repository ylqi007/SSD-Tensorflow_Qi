1. 2018/12/  <br />

* The directory for checkpoints, which contains pretrained checkpoint files.
    ```
    ./checkpoints/
    ```

* The directory for data:   (`./` refers to `SSD-TensorFlow_Qi/`)   <br />
    ```
    ./data/
        |--> UNT_Aerial_Dataset
                |--> test
                |--> train
        |--> VOC2007
                |--> test
                |--> train
    ```
* The directory for package `./datasets/`
    ```
    ./dataset/
        |--> dataset_factory.py
        |--> dataset_utils.py
        |--> pascalvoc_2007.py
        |--> pascalvoc_common.py
        |--> pascalvoc_to_tfrecord.py
        |--> untAerial_to_tfrecord.py
    ```


