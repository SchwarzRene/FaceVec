��
��
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
E
Relu
features"T
activations"T"
Ttype:
2	
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
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
x
Adam/race/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/race/bias/v
q
$Adam/race/bias/v/Read/ReadVariableOpReadVariableOpAdam/race/bias/v*
_output_shapes
:*
dtype0
�
Adam/race/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameAdam/race/kernel/v
z
&Adam/race/kernel/v/Read/ReadVariableOpReadVariableOpAdam/race/kernel/v*
_output_shapes
:	�*
dtype0
v
Adam/age/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:t* 
shared_nameAdam/age/bias/v
o
#Adam/age/bias/v/Read/ReadVariableOpReadVariableOpAdam/age/bias/v*
_output_shapes
:t*
dtype0

Adam/age/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�t*"
shared_nameAdam/age/kernel/v
x
%Adam/age/kernel/v/Read/ReadVariableOpReadVariableOpAdam/age/kernel/v*
_output_shapes
:	�t*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
��*
dtype0
x
Adam/race/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/race/bias/m
q
$Adam/race/bias/m/Read/ReadVariableOpReadVariableOpAdam/race/bias/m*
_output_shapes
:*
dtype0
�
Adam/race/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*#
shared_nameAdam/race/kernel/m
z
&Adam/race/kernel/m/Read/ReadVariableOpReadVariableOpAdam/race/kernel/m*
_output_shapes
:	�*
dtype0
v
Adam/age/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:t* 
shared_nameAdam/age/bias/m
o
#Adam/age/bias/m/Read/ReadVariableOpReadVariableOpAdam/age/bias/m*
_output_shapes
:t*
dtype0

Adam/age/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�t*"
shared_nameAdam/age/kernel/m
x
%Adam/age/kernel/m/Read/ReadVariableOpReadVariableOpAdam/age/kernel/m*
_output_shapes
:	�t*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
��*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
	race/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	race/bias
c
race/bias/Read/ReadVariableOpReadVariableOp	race/bias*
_output_shapes
:*
dtype0
s
race/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namerace/kernel
l
race/kernel/Read/ReadVariableOpReadVariableOprace/kernel*
_output_shapes
:	�*
dtype0
h
age/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_name
age/bias
a
age/bias/Read/ReadVariableOpReadVariableOpage/bias*
_output_shapes
:t*
dtype0
q

age/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�t*
shared_name
age/kernel
j
age/kernel/Read/ReadVariableOpReadVariableOp
age/kernel*
_output_shapes
:	�t*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
��*
dtype0
|
serving_default_input_1Placeholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasrace/kernel	race/bias
age/kernelage/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_506475

NoOpNoOp
�0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�0
value�/B�/ B�/
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
.
0
1
2
3
$4
%5*
.
0
1
2
3
$4
%5*
* 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 
�
3iter

4beta_1

5beta_2
	6decay
7learning_ratem]m^m_m`$ma%mbvcvdvevf$vg%vh*

8serving_default* 

0
1*

0
1*
* 
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

>trace_0* 

?trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Etrace_0* 

Ftrace_0* 
ZT
VARIABLE_VALUE
age/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEage/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ltrace_0* 

Mtrace_0* 
[U
VARIABLE_VALUErace/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	race/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

N0
O1
P2*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
8
Q	variables
R	keras_api
	Stotal
	Tcount*
8
U	variables
V	keras_api
	Wtotal
	Xcount*
8
Y	variables
Z	keras_api
	[total
	\count*

S0
T1*

Q	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

W0
X1*

U	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

Y	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/age/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/age/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/race/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/race/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/age/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/age/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/race/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/race/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpage/kernel/Read/ReadVariableOpage/bias/Read/ReadVariableOprace/kernel/Read/ReadVariableOprace/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp%Adam/age/kernel/m/Read/ReadVariableOp#Adam/age/bias/m/Read/ReadVariableOp&Adam/race/kernel/m/Read/ReadVariableOp$Adam/race/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp%Adam/age/kernel/v/Read/ReadVariableOp#Adam/age/bias/v/Read/ReadVariableOp&Adam/race/kernel/v/Read/ReadVariableOp$Adam/race/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_506736
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias
age/kernelage/biasrace/kernel	race/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/age/kernel/mAdam/age/bias/mAdam/race/kernel/mAdam/race/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/age/kernel/vAdam/age/bias/vAdam/race/kernel/vAdam/race/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_506833��
�
�
A__inference_model_layer_call_and_return_conditional_losses_506565

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�6
#race_matmul_readvariableop_resource:	�2
$race_biasadd_readvariableop_resource:5
"age_matmul_readvariableop_resource:	�t1
#age_biasadd_readvariableop_resource:t
identity

identity_1��age/BiasAdd/ReadVariableOp�age/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�race/BiasAdd/ReadVariableOp�race/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������
race/MatMul/ReadVariableOpReadVariableOp#race_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
race/MatMulMatMuldense/Relu:activations:0"race/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
race/BiasAdd/ReadVariableOpReadVariableOp$race_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
race/BiasAddBiasAddrace/MatMul:product:0#race/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
race/SoftmaxSoftmaxrace/BiasAdd:output:0*
T0*'
_output_shapes
:���������}
age/MatMul/ReadVariableOpReadVariableOp"age_matmul_readvariableop_resource*
_output_shapes
:	�t*
dtype0�

age/MatMulMatMuldense/Relu:activations:0!age/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tz
age/BiasAdd/ReadVariableOpReadVariableOp#age_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0�
age/BiasAddBiasAddage/MatMul:product:0"age/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t^
age/SoftmaxSoftmaxage/BiasAdd:output:0*
T0*'
_output_shapes
:���������td
IdentityIdentityage/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������tg

Identity_1Identityrace/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/BiasAdd/ReadVariableOp^age/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^race/BiasAdd/ReadVariableOp^race/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 28
age/BiasAdd/ReadVariableOpage/BiasAdd/ReadVariableOp26
age/MatMul/ReadVariableOpage/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2:
race/BiasAdd/ReadVariableOprace/BiasAdd/ReadVariableOp28
race/MatMul/ReadVariableOprace/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
&__inference_model_layer_call_fn_506408
input_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:	�t
	unknown_4:t
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_506372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
&__inference_dense_layer_call_fn_506574

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506242p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
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
�
�
A__inference_model_layer_call_and_return_conditional_losses_506284

inputs 
dense_506243:
��
dense_506245:	�
race_506260:	�
race_506262:

age_506277:	�t

age_506279:t
identity

identity_1��age/StatefulPartitionedCall�dense/StatefulPartitionedCall�race/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_506243dense_506245*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506242�
race/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0race_506260race_506262*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_race_layer_call_and_return_conditional_losses_506259�
age/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0
age_506277
age_506279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_age_layer_call_and_return_conditional_losses_506276s
IdentityIdentity$age/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tv

Identity_1Identity%race/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/StatefulPartitionedCall^dense/StatefulPartitionedCall^race/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2:
age/StatefulPartitionedCallage/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
race/StatefulPartitionedCallrace/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
&__inference_model_layer_call_fn_506301
input_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:	�t
	unknown_4:t
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_506284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
A__inference_model_layer_call_and_return_conditional_losses_506428
input_1 
dense_506411:
��
dense_506413:	�
race_506416:	�
race_506418:

age_506421:	�t

age_506423:t
identity

identity_1��age/StatefulPartitionedCall�dense/StatefulPartitionedCall�race/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_506411dense_506413*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506242�
race/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0race_506416race_506418*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_race_layer_call_and_return_conditional_losses_506259�
age/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0
age_506421
age_506423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_age_layer_call_and_return_conditional_losses_506276s
IdentityIdentity$age/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tv

Identity_1Identity%race/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/StatefulPartitionedCall^dense/StatefulPartitionedCall^race/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2:
age/StatefulPartitionedCallage/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
race/StatefulPartitionedCallrace/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�

�
?__inference_age_layer_call_and_return_conditional_losses_506605

inputs1
matmul_readvariableop_resource:	�t-
biasadd_readvariableop_resource:t
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�t*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������t`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_506585

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
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
�

�
$__inference_signature_wrapper_506475
input_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:	�t
	unknown_4:t
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_506224o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�r
�
"__inference__traced_restore_506833
file_prefix1
assignvariableop_dense_kernel:
��,
assignvariableop_1_dense_bias:	�0
assignvariableop_2_age_kernel:	�t)
assignvariableop_3_age_bias:t1
assignvariableop_4_race_kernel:	�*
assignvariableop_5_race_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_2: %
assignvariableop_12_count_2: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: ;
'assignvariableop_17_adam_dense_kernel_m:
��4
%assignvariableop_18_adam_dense_bias_m:	�8
%assignvariableop_19_adam_age_kernel_m:	�t1
#assignvariableop_20_adam_age_bias_m:t9
&assignvariableop_21_adam_race_kernel_m:	�2
$assignvariableop_22_adam_race_bias_m:;
'assignvariableop_23_adam_dense_kernel_v:
��4
%assignvariableop_24_adam_dense_bias_v:	�8
%assignvariableop_25_adam_age_kernel_v:	�t1
#assignvariableop_26_adam_age_bias_v:t9
&assignvariableop_27_adam_race_kernel_v:	�2
$assignvariableop_28_adam_race_bias_v:
identity_30��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_age_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_age_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_race_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_race_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_adam_age_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_adam_age_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_race_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp$assignvariableop_22_adam_race_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_age_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_adam_age_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp&assignvariableop_27_adam_race_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_adam_race_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�

�
&__inference_model_layer_call_fn_506494

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:	�t
	unknown_4:t
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_506284o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_506539

inputs8
$dense_matmul_readvariableop_resource:
��4
%dense_biasadd_readvariableop_resource:	�6
#race_matmul_readvariableop_resource:	�2
$race_biasadd_readvariableop_resource:5
"age_matmul_readvariableop_resource:	�t1
#age_biasadd_readvariableop_resource:t
identity

identity_1��age/BiasAdd/ReadVariableOp�age/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�race/BiasAdd/ReadVariableOp�race/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������
race/MatMul/ReadVariableOpReadVariableOp#race_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
race/MatMulMatMuldense/Relu:activations:0"race/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������|
race/BiasAdd/ReadVariableOpReadVariableOp$race_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
race/BiasAddBiasAddrace/MatMul:product:0#race/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������`
race/SoftmaxSoftmaxrace/BiasAdd:output:0*
T0*'
_output_shapes
:���������}
age/MatMul/ReadVariableOpReadVariableOp"age_matmul_readvariableop_resource*
_output_shapes
:	�t*
dtype0�

age/MatMulMatMuldense/Relu:activations:0!age/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tz
age/BiasAdd/ReadVariableOpReadVariableOp#age_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0�
age/BiasAddBiasAddage/MatMul:product:0"age/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t^
age/SoftmaxSoftmaxage/BiasAdd:output:0*
T0*'
_output_shapes
:���������td
IdentityIdentityage/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������tg

Identity_1Identityrace/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/BiasAdd/ReadVariableOp^age/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^race/BiasAdd/ReadVariableOp^race/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 28
age/BiasAdd/ReadVariableOpage/BiasAdd/ReadVariableOp26
age/MatMul/ReadVariableOpage/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2:
race/BiasAdd/ReadVariableOprace/BiasAdd/ReadVariableOp28
race/MatMul/ReadVariableOprace/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_race_layer_call_fn_506614

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_race_layer_call_and_return_conditional_losses_506259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_506448
input_1 
dense_506431:
��
dense_506433:	�
race_506436:	�
race_506438:

age_506441:	�t

age_506443:t
identity

identity_1��age/StatefulPartitionedCall�dense/StatefulPartitionedCall�race/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_506431dense_506433*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506242�
race/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0race_506436race_506438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_race_layer_call_and_return_conditional_losses_506259�
age/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0
age_506441
age_506443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_age_layer_call_and_return_conditional_losses_506276s
IdentityIdentity$age/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tv

Identity_1Identity%race/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/StatefulPartitionedCall^dense/StatefulPartitionedCall^race/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2:
age/StatefulPartitionedCallage/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
race/StatefulPartitionedCallrace/StatefulPartitionedCall:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
!__inference__wrapped_model_506224
input_1>
*model_dense_matmul_readvariableop_resource:
��:
+model_dense_biasadd_readvariableop_resource:	�<
)model_race_matmul_readvariableop_resource:	�8
*model_race_biasadd_readvariableop_resource:;
(model_age_matmul_readvariableop_resource:	�t7
)model_age_biasadd_readvariableop_resource:t
identity

identity_1�� model/age/BiasAdd/ReadVariableOp�model/age/MatMul/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�!model/race/BiasAdd/ReadVariableOp� model/race/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/dense/MatMulMatMulinput_1)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
 model/race/MatMul/ReadVariableOpReadVariableOp)model_race_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/race/MatMulMatMulmodel/dense/Relu:activations:0(model/race/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!model/race/BiasAdd/ReadVariableOpReadVariableOp*model_race_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/race/BiasAddBiasAddmodel/race/MatMul:product:0)model/race/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������l
model/race/SoftmaxSoftmaxmodel/race/BiasAdd:output:0*
T0*'
_output_shapes
:����������
model/age/MatMul/ReadVariableOpReadVariableOp(model_age_matmul_readvariableop_resource*
_output_shapes
:	�t*
dtype0�
model/age/MatMulMatMulmodel/dense/Relu:activations:0'model/age/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t�
 model/age/BiasAdd/ReadVariableOpReadVariableOp)model_age_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype0�
model/age/BiasAddBiasAddmodel/age/MatMul:product:0(model/age/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tj
model/age/SoftmaxSoftmaxmodel/age/BiasAdd:output:0*
T0*'
_output_shapes
:���������tj
IdentityIdentitymodel/age/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������tm

Identity_1Identitymodel/race/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^model/age/BiasAdd/ReadVariableOp ^model/age/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp"^model/race/BiasAdd/ReadVariableOp!^model/race/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2D
 model/age/BiasAdd/ReadVariableOp model/age/BiasAdd/ReadVariableOp2B
model/age/MatMul/ReadVariableOpmodel/age/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2F
!model/race/BiasAdd/ReadVariableOp!model/race/BiasAdd/ReadVariableOp2D
 model/race/MatMul/ReadVariableOp model/race/MatMul/ReadVariableOp:Q M
(
_output_shapes
:����������
!
_user_specified_name	input_1
�
�
A__inference_model_layer_call_and_return_conditional_losses_506372

inputs 
dense_506355:
��
dense_506357:	�
race_506360:	�
race_506362:

age_506365:	�t

age_506367:t
identity

identity_1��age/StatefulPartitionedCall�dense/StatefulPartitionedCall�race/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_506355dense_506357*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_506242�
race/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0race_506360race_506362*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *I
fDRB
@__inference_race_layer_call_and_return_conditional_losses_506259�
age/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0
age_506365
age_506367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_age_layer_call_and_return_conditional_losses_506276s
IdentityIdentity$age/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tv

Identity_1Identity%race/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^age/StatefulPartitionedCall^dense/StatefulPartitionedCall^race/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 2:
age/StatefulPartitionedCallage/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
race/StatefulPartitionedCallrace/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_race_layer_call_and_return_conditional_losses_506259

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_506242

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
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
�

�
&__inference_model_layer_call_fn_506513

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:
	unknown_3:	�t
	unknown_4:t
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������t:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_506372o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������tq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�=
�
__inference__traced_save_506736
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop)
%savev2_age_kernel_read_readvariableop'
#savev2_age_bias_read_readvariableop*
&savev2_race_kernel_read_readvariableop(
$savev2_race_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop0
,savev2_adam_age_kernel_m_read_readvariableop.
*savev2_adam_age_bias_m_read_readvariableop1
-savev2_adam_race_kernel_m_read_readvariableop/
+savev2_adam_race_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop0
,savev2_adam_age_kernel_v_read_readvariableop.
*savev2_adam_age_bias_v_read_readvariableop1
-savev2_adam_race_kernel_v_read_readvariableop/
+savev2_adam_race_bias_v_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop%savev2_age_kernel_read_readvariableop#savev2_age_bias_read_readvariableop&savev2_race_kernel_read_readvariableop$savev2_race_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop,savev2_adam_age_kernel_m_read_readvariableop*savev2_adam_age_bias_m_read_readvariableop-savev2_adam_race_kernel_m_read_readvariableop+savev2_adam_race_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop,savev2_adam_age_kernel_v_read_readvariableop*savev2_adam_age_bias_v_read_readvariableop-savev2_adam_race_kernel_v_read_readvariableop+savev2_adam_race_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:	�t:t:	�:: : : : : : : : : : : :
��:�:	�t:t:	�::
��:�:	�t:t:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�t: 

_output_shapes
:t:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�t: 

_output_shapes
:t:%!

_output_shapes
:	�: 

_output_shapes
::&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�t: 

_output_shapes
:t:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
$__inference_age_layer_call_fn_506594

inputs
unknown:	�t
	unknown_0:t
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_age_layer_call_and_return_conditional_losses_506276o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������t`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
?__inference_age_layer_call_and_return_conditional_losses_506276

inputs1
matmul_readvariableop_resource:	�t-
biasadd_readvariableop_resource:t
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�t*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������tV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������t`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������tw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
@__inference_race_layer_call_and_return_conditional_losses_506625

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
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
input_11
serving_default_input_1:0����������7
age0
StatefulPartitionedCall:0���������t8
race0
StatefulPartitionedCall:1���������tensorflow/serving/predict:�p
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
J
0
1
2
3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
+trace_0
,trace_1
-trace_2
.trace_32�
&__inference_model_layer_call_fn_506301
&__inference_model_layer_call_fn_506494
&__inference_model_layer_call_fn_506513
&__inference_model_layer_call_fn_506408�
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
 z+trace_0z,trace_1z-trace_2z.trace_3
�
/trace_0
0trace_1
1trace_2
2trace_32�
A__inference_model_layer_call_and_return_conditional_losses_506539
A__inference_model_layer_call_and_return_conditional_losses_506565
A__inference_model_layer_call_and_return_conditional_losses_506428
A__inference_model_layer_call_and_return_conditional_losses_506448�
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
 z/trace_0z0trace_1z1trace_2z2trace_3
�B�
!__inference__wrapped_model_506224input_1"�
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
�
3iter

4beta_1

5beta_2
	6decay
7learning_ratem]m^m_m`$ma%mbvcvdvevf$vg%vh"
	optimizer
,
8serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
>trace_02�
&__inference_dense_layer_call_fn_506574�
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
 z>trace_0
�
?trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_506585�
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
 z?trace_0
 :
��2dense/kernel
:�2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Etrace_02�
$__inference_age_layer_call_fn_506594�
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
 zEtrace_0
�
Ftrace_02�
?__inference_age_layer_call_and_return_conditional_losses_506605�
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
 zFtrace_0
:	�t2
age/kernel
:t2age/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Ltrace_02�
%__inference_race_layer_call_fn_506614�
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
 zLtrace_0
�
Mtrace_02�
@__inference_race_layer_call_and_return_conditional_losses_506625�
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
 zMtrace_0
:	�2race/kernel
:2	race/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_model_layer_call_fn_506301input_1"�
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
&__inference_model_layer_call_fn_506494inputs"�
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
&__inference_model_layer_call_fn_506513inputs"�
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
&__inference_model_layer_call_fn_506408input_1"�
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
A__inference_model_layer_call_and_return_conditional_losses_506539inputs"�
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
A__inference_model_layer_call_and_return_conditional_losses_506565inputs"�
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
A__inference_model_layer_call_and_return_conditional_losses_506428input_1"�
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
A__inference_model_layer_call_and_return_conditional_losses_506448input_1"�
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
$__inference_signature_wrapper_506475input_1"�
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
&__inference_dense_layer_call_fn_506574inputs"�
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
A__inference_dense_layer_call_and_return_conditional_losses_506585inputs"�
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
$__inference_age_layer_call_fn_506594inputs"�
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
?__inference_age_layer_call_and_return_conditional_losses_506605inputs"�
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
%__inference_race_layer_call_fn_506614inputs"�
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
@__inference_race_layer_call_and_return_conditional_losses_506625inputs"�
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
N
Q	variables
R	keras_api
	Stotal
	Tcount"
_tf_keras_metric
N
U	variables
V	keras_api
	Wtotal
	Xcount"
_tf_keras_metric
N
Y	variables
Z	keras_api
	[total
	\count"
_tf_keras_metric
.
S0
T1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
.
W0
X1"
trackable_list_wrapper
-
U	variables"
_generic_user_object
:  (2total
:  (2count
.
[0
\1"
trackable_list_wrapper
-
Y	variables"
_generic_user_object
:  (2total
:  (2count
%:#
��2Adam/dense/kernel/m
:�2Adam/dense/bias/m
": 	�t2Adam/age/kernel/m
:t2Adam/age/bias/m
#:!	�2Adam/race/kernel/m
:2Adam/race/bias/m
%:#
��2Adam/dense/kernel/v
:�2Adam/dense/bias/v
": 	�t2Adam/age/kernel/v
:t2Adam/age/bias/v
#:!	�2Adam/race/kernel/v
:2Adam/race/bias/v�
!__inference__wrapped_model_506224�$%1�.
'�$
"�
input_1����������
� "Q�N
$
age�
age���������t
&
race�
race����������
?__inference_age_layer_call_and_return_conditional_losses_506605]0�-
&�#
!�
inputs����������
� "%�"
�
0���������t
� x
$__inference_age_layer_call_fn_506594P0�-
&�#
!�
inputs����������
� "����������t�
A__inference_dense_layer_call_and_return_conditional_losses_506585^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� {
&__inference_dense_layer_call_fn_506574Q0�-
&�#
!�
inputs����������
� "������������
A__inference_model_layer_call_and_return_conditional_losses_506428�$%9�6
/�,
"�
input_1����������
p 

 
� "K�H
A�>
�
0/0���������t
�
0/1���������
� �
A__inference_model_layer_call_and_return_conditional_losses_506448�$%9�6
/�,
"�
input_1����������
p

 
� "K�H
A�>
�
0/0���������t
�
0/1���������
� �
A__inference_model_layer_call_and_return_conditional_losses_506539�$%8�5
.�+
!�
inputs����������
p 

 
� "K�H
A�>
�
0/0���������t
�
0/1���������
� �
A__inference_model_layer_call_and_return_conditional_losses_506565�$%8�5
.�+
!�
inputs����������
p

 
� "K�H
A�>
�
0/0���������t
�
0/1���������
� �
&__inference_model_layer_call_fn_506301�$%9�6
/�,
"�
input_1����������
p 

 
� "=�:
�
0���������t
�
1����������
&__inference_model_layer_call_fn_506408�$%9�6
/�,
"�
input_1����������
p

 
� "=�:
�
0���������t
�
1����������
&__inference_model_layer_call_fn_506494�$%8�5
.�+
!�
inputs����������
p 

 
� "=�:
�
0���������t
�
1����������
&__inference_model_layer_call_fn_506513�$%8�5
.�+
!�
inputs����������
p

 
� "=�:
�
0���������t
�
1����������
@__inference_race_layer_call_and_return_conditional_losses_506625]$%0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� y
%__inference_race_layer_call_fn_506614P$%0�-
&�#
!�
inputs����������
� "�����������
$__inference_signature_wrapper_506475�$%<�9
� 
2�/
-
input_1"�
input_1����������"Q�N
$
age�
age���������t
&
race�
race���������