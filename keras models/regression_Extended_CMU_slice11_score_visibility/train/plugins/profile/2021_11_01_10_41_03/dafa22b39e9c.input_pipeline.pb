  *	A`����@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapa��
�,$@!E ��%fW@)�?�@��#@1����1W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_vO��?!�����@)�HLP÷�?1� ��@:Preprocessing2�
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate[0]::TensorSlice�2�FY�?!�p,P�?)�2�FY�?1�p,P�?:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�?4�䚶?!�&Jw�7�?)�p�Ws��?1$RB�C��?:Preprocessing2F
Iterator::Model�.�e���?!��>����?)y���h�?1SIxݵO�?:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::Concatenate��։��?!h6�5@�?) �o_Ι?1�i�+���?:Preprocessing2U
Iterator::Model::ParallelMapV2�UIdd�?!�
�r�?)�UIdd�?1�
�r�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatr��Q���?!��d����?)��1zn�?1�])�7�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::PrefetchC��fڎ?!���]��?)C��fڎ?1���]��?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��FXT��?!� ���@)Y�8��m�?1�s�"���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~NA~6�?!w5���?)�~NA~6�?1w5���?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�unڌӀ?!�>���?)�unڌӀ?1�>���?:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate�f׽�?!�_~���?)�4�($i?1�Eˌ�(�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::Concatenate[1]::FromTensors�m�B<R?!�}�Y&�?)s�m�B<R?1�}�Y&�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.