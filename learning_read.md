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

2. How to create tfrecord files and read data from tfrecord files?
    [tfrecord files](http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html)
    * Get data from tfrecord files:
        ```
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)

        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox'])
        ```

3. Create SSD network.
* The directory for nets:
    ```
    ./nets/
        |--> custom_layers.py
        |--> net_factory.py
        |--> ssd_vgg_300.py
    ```

    1. ssd_vgg_300.SSDNet.net()
    2. ssd_vgg_300.SSDNet.anchors()
    3. ssd_common.tf_ssd_bboxes_encode() ==> Not check yet, because it's complex to get groundtruth boxes.

