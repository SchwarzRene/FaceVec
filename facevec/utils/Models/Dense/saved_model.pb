��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�O*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:�O*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��O*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
��O*
dtype0
|
serving_default_input_2Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_237210

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories*

0
1*

0
1*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
 trace_1
!trace_2
"trace_3* 
* 

#serving_default* 
* 
* 

0
1*

0
1*
* 
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

)trace_0* 

*trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_237299
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_237315�
�
�
(__inference_model_1_layer_call_fn_237181
input_2
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_237165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
(__inference_dense_2_layer_call_fn_237259

inputs
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_237121p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_237219

inputs
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_237128p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237250

inputs:
&dense_2_matmul_readvariableop_resource:
��O6
'dense_2_biasadd_readvariableop_resource:	�O
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��O*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�O*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Og
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������Oi
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������O�
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_237228

inputs
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_237165p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference__traced_restore_237315
file_prefix3
assignvariableop_dense_2_kernel:
��O.
assignvariableop_1_dense_2_bias:	�O

identity_3��AssignVariableOp�AssignVariableOp_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: p
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_model_1_layer_call_fn_237135
input_2
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_237128p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237165

inputs"
dense_2_237159:
��O
dense_2_237161:	�O
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_237159dense_2_237161*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_237121x
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������Oh
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237190
input_2"
dense_2_237184:
��O
dense_2_237186:	�O
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_237184dense_2_237186*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_237121x
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������Oh
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_237270

inputs2
matmul_readvariableop_resource:
��O.
biasadd_readvariableop_resource:	�O
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��O*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Os
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�O*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������OW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������Oa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������Ow
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference__traced_save_237299
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0**
_input_shapes
: :
��O:�O: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��O:!

_output_shapes	
:�O:

_output_shapes
: 
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237128

inputs"
dense_2_237122:
��O
dense_2_237124:	�O
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_237122dense_2_237124*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_237121x
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������Oh
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237199
input_2"
dense_2_237193:
��O
dense_2_237195:	�O
identity��dense_2/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_237193dense_2_237195*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_237121x
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������Oh
NoOpNoOp ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_237121

inputs2
matmul_readvariableop_resource:
��O.
biasadd_readvariableop_resource:	�O
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��O*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Os
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�O*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������OW
SoftmaxSoftmaxBiasAdd:output:0*
T0*(
_output_shapes
:����������Oa
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������Ow
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_237210
input_2
unknown:
��O
	unknown_0:	�O
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������O*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_237103p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������O`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
!__inference__wrapped_model_237103
input_2B
.model_1_dense_2_matmul_readvariableop_resource:
��O>
/model_1_dense_2_biasadd_readvariableop_resource:	�O
identity��&model_1/dense_2/BiasAdd/ReadVariableOp�%model_1/dense_2/MatMul/ReadVariableOp�
%model_1/dense_2/MatMul/ReadVariableOpReadVariableOp.model_1_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��O*
dtype0�
model_1/dense_2/MatMulMatMulinput_2-model_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O�
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�O*
dtype0�
model_1/dense_2/BiasAddBiasAdd model_1/dense_2/MatMul:product:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Ow
model_1/dense_2/SoftmaxSoftmax model_1/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������Oq
IdentityIdentity!model_1/dense_2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������O�
NoOpNoOp'^model_1/dense_2/BiasAdd/ReadVariableOp&^model_1/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2N
%model_1/dense_2/MatMul/ReadVariableOp%model_1/dense_2/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_2
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_237239

inputs:
&dense_2_matmul_readvariableop_resource:
��O6
'dense_2_biasadd_readvariableop_resource:	�O
identity��dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��O*
dtype0z
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������O�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�O*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Og
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������Oi
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*(
_output_shapes
:����������O�
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
<
input_21
serving_default_input_2:0����������<
dense_21
StatefulPartitionedCall:0����������Otensorflow/serving/predict:�B
�
layer-0
layer_with_weights-0
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature


signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
#_self_saveable_object_factories"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_1
trace_2
trace_32�
(__inference_model_1_layer_call_fn_237135
(__inference_model_1_layer_call_fn_237219
(__inference_model_1_layer_call_fn_237228
(__inference_model_1_layer_call_fn_237181�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1ztrace_2ztrace_3
�
trace_0
 trace_1
!trace_2
"trace_32�
C__inference_model_1_layer_call_and_return_conditional_losses_237239
C__inference_model_1_layer_call_and_return_conditional_losses_237250
C__inference_model_1_layer_call_and_return_conditional_losses_237190
C__inference_model_1_layer_call_and_return_conditional_losses_237199�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0z trace_1z!trace_2z"trace_3
�B�
!__inference__wrapped_model_237103input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
#serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
)trace_02�
(__inference_dense_2_layer_call_fn_237259�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z)trace_0
�
*trace_02�
C__inference_dense_2_layer_call_and_return_conditional_losses_237270�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z*trace_0
": 
��O2dense_2/kernel
:�O2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_1_layer_call_fn_237135input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_237219inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_237228inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_1_layer_call_fn_237181input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_237239inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_237250inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_237190input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_1_layer_call_and_return_conditional_losses_237199input_2"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_237210input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_2_layer_call_fn_237259inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_2_layer_call_and_return_conditional_losses_237270inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_237103k1�.
'�$
"�
input_2����������
� "2�/
-
dense_2"�
dense_2����������O�
C__inference_dense_2_layer_call_and_return_conditional_losses_237270^0�-
&�#
!�
inputs����������
� "&�#
�
0����������O
� }
(__inference_dense_2_layer_call_fn_237259Q0�-
&�#
!�
inputs����������
� "�����������O�
C__inference_model_1_layer_call_and_return_conditional_losses_237190g9�6
/�,
"�
input_2����������
p 

 
� "&�#
�
0����������O
� �
C__inference_model_1_layer_call_and_return_conditional_losses_237199g9�6
/�,
"�
input_2����������
p

 
� "&�#
�
0����������O
� �
C__inference_model_1_layer_call_and_return_conditional_losses_237239f8�5
.�+
!�
inputs����������
p 

 
� "&�#
�
0����������O
� �
C__inference_model_1_layer_call_and_return_conditional_losses_237250f8�5
.�+
!�
inputs����������
p

 
� "&�#
�
0����������O
� �
(__inference_model_1_layer_call_fn_237135Z9�6
/�,
"�
input_2����������
p 

 
� "�����������O�
(__inference_model_1_layer_call_fn_237181Z9�6
/�,
"�
input_2����������
p

 
� "�����������O�
(__inference_model_1_layer_call_fn_237219Y8�5
.�+
!�
inputs����������
p 

 
� "�����������O�
(__inference_model_1_layer_call_fn_237228Y8�5
.�+
!�
inputs����������
p

 
� "�����������O�
$__inference_signature_wrapper_237210v<�9
� 
2�/
-
input_2"�
input_2����������"2�/
-
dense_2"�
dense_2����������O