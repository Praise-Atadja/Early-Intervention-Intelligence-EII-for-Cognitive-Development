��	
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
SGD/m/dense_479/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_479/bias/*
dtype0*
shape:*%
shared_nameSGD/m/dense_479/bias
y
(SGD/m/dense_479/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_479/bias*
_output_shapes
:*
dtype0
�
SGD/m/dense_479/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_479/kernel/*
dtype0*
shape:	�*'
shared_nameSGD/m/dense_479/kernel
�
*SGD/m/dense_479/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_479/kernel*
_output_shapes
:	�*
dtype0
�
SGD/m/dense_478/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_478/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_478/bias
z
(SGD/m/dense_478/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_478/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_478/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_478/kernel/*
dtype0*
shape:
��*'
shared_nameSGD/m/dense_478/kernel
�
*SGD/m/dense_478/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_478/kernel* 
_output_shapes
:
��*
dtype0
�
SGD/m/dense_477/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_477/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_477/bias
z
(SGD/m/dense_477/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_477/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_477/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_477/kernel/*
dtype0*
shape:
��*'
shared_nameSGD/m/dense_477/kernel
�
*SGD/m/dense_477/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_477/kernel* 
_output_shapes
:
��*
dtype0
�
SGD/m/dense_476/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_476/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_476/bias
z
(SGD/m/dense_476/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_476/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_476/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_476/kernel/*
dtype0*
shape:
��*'
shared_nameSGD/m/dense_476/kernel
�
*SGD/m/dense_476/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_476/kernel* 
_output_shapes
:
��*
dtype0
�
SGD/m/dense_475/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_475/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_475/bias
z
(SGD/m/dense_475/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_475/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_475/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_475/kernel/*
dtype0*
shape:
��*'
shared_nameSGD/m/dense_475/kernel
�
*SGD/m/dense_475/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_475/kernel* 
_output_shapes
:
��*
dtype0
�
SGD/m/dense_474/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_474/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_474/bias
z
(SGD/m/dense_474/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_474/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_474/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_474/kernel/*
dtype0*
shape:
��*'
shared_nameSGD/m/dense_474/kernel
�
*SGD/m/dense_474/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_474/kernel* 
_output_shapes
:
��*
dtype0
�
SGD/m/dense_473/biasVarHandleOp*
_output_shapes
: *%

debug_nameSGD/m/dense_473/bias/*
dtype0*
shape:�*%
shared_nameSGD/m/dense_473/bias
z
(SGD/m/dense_473/bias/Read/ReadVariableOpReadVariableOpSGD/m/dense_473/bias*
_output_shapes	
:�*
dtype0
�
SGD/m/dense_473/kernelVarHandleOp*
_output_shapes
: *'

debug_nameSGD/m/dense_473/kernel/*
dtype0*
shape:	�*'
shared_nameSGD/m/dense_473/kernel
�
*SGD/m/dense_473/kernel/Read/ReadVariableOpReadVariableOpSGD/m/dense_473/kernel*
_output_shapes
:	�*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_479/biasVarHandleOp*
_output_shapes
: *

debug_namedense_479/bias/*
dtype0*
shape:*
shared_namedense_479/bias
m
"dense_479/bias/Read/ReadVariableOpReadVariableOpdense_479/bias*
_output_shapes
:*
dtype0
�
dense_479/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_479/kernel/*
dtype0*
shape:	�*!
shared_namedense_479/kernel
v
$dense_479/kernel/Read/ReadVariableOpReadVariableOpdense_479/kernel*
_output_shapes
:	�*
dtype0
�
dense_478/biasVarHandleOp*
_output_shapes
: *

debug_namedense_478/bias/*
dtype0*
shape:�*
shared_namedense_478/bias
n
"dense_478/bias/Read/ReadVariableOpReadVariableOpdense_478/bias*
_output_shapes	
:�*
dtype0
�
dense_478/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_478/kernel/*
dtype0*
shape:
��*!
shared_namedense_478/kernel
w
$dense_478/kernel/Read/ReadVariableOpReadVariableOpdense_478/kernel* 
_output_shapes
:
��*
dtype0
�
dense_477/biasVarHandleOp*
_output_shapes
: *

debug_namedense_477/bias/*
dtype0*
shape:�*
shared_namedense_477/bias
n
"dense_477/bias/Read/ReadVariableOpReadVariableOpdense_477/bias*
_output_shapes	
:�*
dtype0
�
dense_477/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_477/kernel/*
dtype0*
shape:
��*!
shared_namedense_477/kernel
w
$dense_477/kernel/Read/ReadVariableOpReadVariableOpdense_477/kernel* 
_output_shapes
:
��*
dtype0
�
dense_476/biasVarHandleOp*
_output_shapes
: *

debug_namedense_476/bias/*
dtype0*
shape:�*
shared_namedense_476/bias
n
"dense_476/bias/Read/ReadVariableOpReadVariableOpdense_476/bias*
_output_shapes	
:�*
dtype0
�
dense_476/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_476/kernel/*
dtype0*
shape:
��*!
shared_namedense_476/kernel
w
$dense_476/kernel/Read/ReadVariableOpReadVariableOpdense_476/kernel* 
_output_shapes
:
��*
dtype0
�
dense_475/biasVarHandleOp*
_output_shapes
: *

debug_namedense_475/bias/*
dtype0*
shape:�*
shared_namedense_475/bias
n
"dense_475/bias/Read/ReadVariableOpReadVariableOpdense_475/bias*
_output_shapes	
:�*
dtype0
�
dense_475/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_475/kernel/*
dtype0*
shape:
��*!
shared_namedense_475/kernel
w
$dense_475/kernel/Read/ReadVariableOpReadVariableOpdense_475/kernel* 
_output_shapes
:
��*
dtype0
�
dense_474/biasVarHandleOp*
_output_shapes
: *

debug_namedense_474/bias/*
dtype0*
shape:�*
shared_namedense_474/bias
n
"dense_474/bias/Read/ReadVariableOpReadVariableOpdense_474/bias*
_output_shapes	
:�*
dtype0
�
dense_474/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_474/kernel/*
dtype0*
shape:
��*!
shared_namedense_474/kernel
w
$dense_474/kernel/Read/ReadVariableOpReadVariableOpdense_474/kernel* 
_output_shapes
:
��*
dtype0
�
dense_473/biasVarHandleOp*
_output_shapes
: *

debug_namedense_473/bias/*
dtype0*
shape:�*
shared_namedense_473/bias
n
"dense_473/bias/Read/ReadVariableOpReadVariableOpdense_473/bias*
_output_shapes	
:�*
dtype0
�
dense_473/kernelVarHandleOp*
_output_shapes
: *!

debug_namedense_473/kernel/*
dtype0*
shape:	�*!
shared_namedense_473/kernel
v
$dense_473/kernel/Read/ReadVariableOpReadVariableOpdense_473/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_dense_473_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_473_inputdense_473/kerneldense_473/biasdense_474/kerneldense_474/biasdense_475/kerneldense_475/biasdense_476/kerneldense_476/biasdense_477/kerneldense_477/biasdense_478/kerneldense_478/biasdense_479/kerneldense_479/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_555193

NoOpNoOp
�F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�E
value�EB�E B�E
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias*
j
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13*
j
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ntrace_0
Otrace_1* 

Ptrace_0
Qtrace_1* 
* 
o
R
_variables
S_iterations
T_learning_rate
U_index_dict
V	momentums
W_update_step_xla*

Xserving_default* 

0
1*

0
1*
* 
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
`Z
VARIABLE_VALUEdense_473/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_473/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
`Z
VARIABLE_VALUEdense_474/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_474/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 
`Z
VARIABLE_VALUEdense_475/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_475/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
`Z
VARIABLE_VALUEdense_476/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_476/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

ztrace_0* 

{trace_0* 
`Z
VARIABLE_VALUEdense_477/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_477/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_478/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_478/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

G0
H1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_479/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_479/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

�0
�1*
* 
* 
* 
* 
* 
* 
�
S0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13*
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
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
a[
VARIABLE_VALUESGD/m/dense_473/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUESGD/m/dense_473/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUESGD/m/dense_474/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUESGD/m/dense_474/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUESGD/m/dense_475/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUESGD/m/dense_475/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUESGD/m/dense_476/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUESGD/m/dense_476/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUESGD/m/dense_477/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUESGD/m/dense_477/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/dense_478/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUESGD/m/dense_478/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUESGD/m/dense_479/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUESGD/m/dense_479/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_473/kerneldense_473/biasdense_474/kerneldense_474/biasdense_475/kerneldense_475/biasdense_476/kerneldense_476/biasdense_477/kerneldense_477/biasdense_478/kerneldense_478/biasdense_479/kerneldense_479/bias	iterationlearning_rateSGD/m/dense_473/kernelSGD/m/dense_473/biasSGD/m/dense_474/kernelSGD/m/dense_474/biasSGD/m/dense_475/kernelSGD/m/dense_475/biasSGD/m/dense_476/kernelSGD/m/dense_476/biasSGD/m/dense_477/kernelSGD/m/dense_477/biasSGD/m/dense_478/kernelSGD/m/dense_478/biasSGD/m/dense_479/kernelSGD/m/dense_479/biastotal_1count_1totalcountConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_555559
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_473/kerneldense_473/biasdense_474/kerneldense_474/biasdense_475/kerneldense_475/biasdense_476/kerneldense_476/biasdense_477/kerneldense_477/biasdense_478/kerneldense_478/biasdense_479/kerneldense_479/bias	iterationlearning_rateSGD/m/dense_473/kernelSGD/m/dense_473/biasSGD/m/dense_474/kernelSGD/m/dense_474/biasSGD/m/dense_475/kernelSGD/m/dense_475/biasSGD/m/dense_476/kernelSGD/m/dense_476/biasSGD/m/dense_477/kernelSGD/m/dense_477/biasSGD/m/dense_478/kernelSGD/m/dense_478/biasSGD/m/dense_479/kernelSGD/m/dense_479/biastotal_1count_1totalcount*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_555670��
�

�
E__inference_dense_477_layer_call_and_return_conditional_losses_555293

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_dense_473_layer_call_fn_555202

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_473_layer_call_and_return_conditional_losses_554886p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name555196:&"
 
_user_specified_name555198
�*
�
I__inference_sequential_69_layer_call_and_return_conditional_losses_554989
dense_473_input#
dense_473_554887:	�
dense_473_554889:	�$
dense_474_554903:
��
dense_474_554905:	�$
dense_475_554919:
��
dense_475_554921:	�$
dense_476_554935:
��
dense_476_554937:	�$
dense_477_554951:
��
dense_477_554953:	�$
dense_478_554967:
��
dense_478_554969:	�#
dense_479_554983:	�
dense_479_554985:
identity��!dense_473/StatefulPartitionedCall�!dense_474/StatefulPartitionedCall�!dense_475/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
!dense_473/StatefulPartitionedCallStatefulPartitionedCalldense_473_inputdense_473_554887dense_473_554889*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_473_layer_call_and_return_conditional_losses_554886�
!dense_474/StatefulPartitionedCallStatefulPartitionedCall*dense_473/StatefulPartitionedCall:output:0dense_474_554903dense_474_554905*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_474_layer_call_and_return_conditional_losses_554902�
!dense_475/StatefulPartitionedCallStatefulPartitionedCall*dense_474/StatefulPartitionedCall:output:0dense_475_554919dense_475_554921*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_475_layer_call_and_return_conditional_losses_554918�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall*dense_475/StatefulPartitionedCall:output:0dense_476_554935dense_476_554937*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_554934�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_554951dense_477_554953*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_554950�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_554967dense_478_554969*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_554966�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_554983dense_479_554985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_554982y
IdentityIdentity*dense_479/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_473/StatefulPartitionedCall"^dense_474/StatefulPartitionedCall"^dense_475/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2F
!dense_473/StatefulPartitionedCall!dense_473/StatefulPartitionedCall2F
!dense_474/StatefulPartitionedCall!dense_474/StatefulPartitionedCall2F
!dense_475/StatefulPartitionedCall!dense_475/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:&"
 
_user_specified_name554887:&"
 
_user_specified_name554889:&"
 
_user_specified_name554903:&"
 
_user_specified_name554905:&"
 
_user_specified_name554919:&"
 
_user_specified_name554921:&"
 
_user_specified_name554935:&"
 
_user_specified_name554937:&	"
 
_user_specified_name554951:&
"
 
_user_specified_name554953:&"
 
_user_specified_name554967:&"
 
_user_specified_name554969:&"
 
_user_specified_name554983:&"
 
_user_specified_name554985
�*
�
I__inference_sequential_69_layer_call_and_return_conditional_losses_555028
dense_473_input#
dense_473_554992:	�
dense_473_554994:	�$
dense_474_554997:
��
dense_474_554999:	�$
dense_475_555002:
��
dense_475_555004:	�$
dense_476_555007:
��
dense_476_555009:	�$
dense_477_555012:
��
dense_477_555014:	�$
dense_478_555017:
��
dense_478_555019:	�#
dense_479_555022:	�
dense_479_555024:
identity��!dense_473/StatefulPartitionedCall�!dense_474/StatefulPartitionedCall�!dense_475/StatefulPartitionedCall�!dense_476/StatefulPartitionedCall�!dense_477/StatefulPartitionedCall�!dense_478/StatefulPartitionedCall�!dense_479/StatefulPartitionedCall�
!dense_473/StatefulPartitionedCallStatefulPartitionedCalldense_473_inputdense_473_554992dense_473_554994*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_473_layer_call_and_return_conditional_losses_554886�
!dense_474/StatefulPartitionedCallStatefulPartitionedCall*dense_473/StatefulPartitionedCall:output:0dense_474_554997dense_474_554999*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_474_layer_call_and_return_conditional_losses_554902�
!dense_475/StatefulPartitionedCallStatefulPartitionedCall*dense_474/StatefulPartitionedCall:output:0dense_475_555002dense_475_555004*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_475_layer_call_and_return_conditional_losses_554918�
!dense_476/StatefulPartitionedCallStatefulPartitionedCall*dense_475/StatefulPartitionedCall:output:0dense_476_555007dense_476_555009*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_554934�
!dense_477/StatefulPartitionedCallStatefulPartitionedCall*dense_476/StatefulPartitionedCall:output:0dense_477_555012dense_477_555014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_554950�
!dense_478/StatefulPartitionedCallStatefulPartitionedCall*dense_477/StatefulPartitionedCall:output:0dense_478_555017dense_478_555019*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_554966�
!dense_479/StatefulPartitionedCallStatefulPartitionedCall*dense_478/StatefulPartitionedCall:output:0dense_479_555022dense_479_555024*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_554982y
IdentityIdentity*dense_479/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_473/StatefulPartitionedCall"^dense_474/StatefulPartitionedCall"^dense_475/StatefulPartitionedCall"^dense_476/StatefulPartitionedCall"^dense_477/StatefulPartitionedCall"^dense_478/StatefulPartitionedCall"^dense_479/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2F
!dense_473/StatefulPartitionedCall!dense_473/StatefulPartitionedCall2F
!dense_474/StatefulPartitionedCall!dense_474/StatefulPartitionedCall2F
!dense_475/StatefulPartitionedCall!dense_475/StatefulPartitionedCall2F
!dense_476/StatefulPartitionedCall!dense_476/StatefulPartitionedCall2F
!dense_477/StatefulPartitionedCall!dense_477/StatefulPartitionedCall2F
!dense_478/StatefulPartitionedCall!dense_478/StatefulPartitionedCall2F
!dense_479/StatefulPartitionedCall!dense_479/StatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:&"
 
_user_specified_name554992:&"
 
_user_specified_name554994:&"
 
_user_specified_name554997:&"
 
_user_specified_name554999:&"
 
_user_specified_name555002:&"
 
_user_specified_name555004:&"
 
_user_specified_name555007:&"
 
_user_specified_name555009:&	"
 
_user_specified_name555012:&
"
 
_user_specified_name555014:&"
 
_user_specified_name555017:&"
 
_user_specified_name555019:&"
 
_user_specified_name555022:&"
 
_user_specified_name555024
�

�
E__inference_dense_474_layer_call_and_return_conditional_losses_555233

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_473_layer_call_and_return_conditional_losses_554886

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_dense_479_layer_call_fn_555322

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_479_layer_call_and_return_conditional_losses_554982o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555316:&"
 
_user_specified_name555318
�

�
E__inference_dense_478_layer_call_and_return_conditional_losses_555313

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_dense_478_layer_call_fn_555302

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_478_layer_call_and_return_conditional_losses_554966p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555296:&"
 
_user_specified_name555298
�

�
E__inference_dense_477_layer_call_and_return_conditional_losses_554950

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
"__inference__traced_restore_555670
file_prefix4
!assignvariableop_dense_473_kernel:	�0
!assignvariableop_1_dense_473_bias:	�7
#assignvariableop_2_dense_474_kernel:
��0
!assignvariableop_3_dense_474_bias:	�7
#assignvariableop_4_dense_475_kernel:
��0
!assignvariableop_5_dense_475_bias:	�7
#assignvariableop_6_dense_476_kernel:
��0
!assignvariableop_7_dense_476_bias:	�7
#assignvariableop_8_dense_477_kernel:
��0
!assignvariableop_9_dense_477_bias:	�8
$assignvariableop_10_dense_478_kernel:
��1
"assignvariableop_11_dense_478_bias:	�7
$assignvariableop_12_dense_479_kernel:	�0
"assignvariableop_13_dense_479_bias:'
assignvariableop_14_iteration:	 +
!assignvariableop_15_learning_rate: =
*assignvariableop_16_sgd_m_dense_473_kernel:	�7
(assignvariableop_17_sgd_m_dense_473_bias:	�>
*assignvariableop_18_sgd_m_dense_474_kernel:
��7
(assignvariableop_19_sgd_m_dense_474_bias:	�>
*assignvariableop_20_sgd_m_dense_475_kernel:
��7
(assignvariableop_21_sgd_m_dense_475_bias:	�>
*assignvariableop_22_sgd_m_dense_476_kernel:
��7
(assignvariableop_23_sgd_m_dense_476_bias:	�>
*assignvariableop_24_sgd_m_dense_477_kernel:
��7
(assignvariableop_25_sgd_m_dense_477_bias:	�>
*assignvariableop_26_sgd_m_dense_478_kernel:
��7
(assignvariableop_27_sgd_m_dense_478_bias:	�=
*assignvariableop_28_sgd_m_dense_479_kernel:	�6
(assignvariableop_29_sgd_m_dense_479_bias:%
assignvariableop_30_total_1: %
assignvariableop_31_count_1: #
assignvariableop_32_total: #
assignvariableop_33_count: 
identity_35��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dense_473_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_473_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_474_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_474_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_475_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_475_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_476_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_476_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_477_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_477_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_478_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_478_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dense_479_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dense_479_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_iterationIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_sgd_m_dense_473_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_sgd_m_dense_473_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp*assignvariableop_18_sgd_m_dense_474_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_sgd_m_dense_474_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp*assignvariableop_20_sgd_m_dense_475_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp(assignvariableop_21_sgd_m_dense_475_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_sgd_m_dense_476_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_sgd_m_dense_476_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp*assignvariableop_24_sgd_m_dense_477_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp(assignvariableop_25_sgd_m_dense_477_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_sgd_m_dense_478_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp(assignvariableop_27_sgd_m_dense_478_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp*assignvariableop_28_sgd_m_dense_479_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_sgd_m_dense_479_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_1Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_1Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_totalIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_countIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_35Identity_35:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
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
_user_specified_namefile_prefix:0,
*
_user_specified_namedense_473/kernel:.*
(
_user_specified_namedense_473/bias:0,
*
_user_specified_namedense_474/kernel:.*
(
_user_specified_namedense_474/bias:0,
*
_user_specified_namedense_475/kernel:.*
(
_user_specified_namedense_475/bias:0,
*
_user_specified_namedense_476/kernel:.*
(
_user_specified_namedense_476/bias:0	,
*
_user_specified_namedense_477/kernel:.
*
(
_user_specified_namedense_477/bias:0,
*
_user_specified_namedense_478/kernel:.*
(
_user_specified_namedense_478/bias:0,
*
_user_specified_namedense_479/kernel:.*
(
_user_specified_namedense_479/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:62
0
_user_specified_nameSGD/m/dense_473/kernel:40
.
_user_specified_nameSGD/m/dense_473/bias:62
0
_user_specified_nameSGD/m/dense_474/kernel:40
.
_user_specified_nameSGD/m/dense_474/bias:62
0
_user_specified_nameSGD/m/dense_475/kernel:40
.
_user_specified_nameSGD/m/dense_475/bias:62
0
_user_specified_nameSGD/m/dense_476/kernel:40
.
_user_specified_nameSGD/m/dense_476/bias:62
0
_user_specified_nameSGD/m/dense_477/kernel:40
.
_user_specified_nameSGD/m/dense_477/bias:62
0
_user_specified_nameSGD/m/dense_478/kernel:40
.
_user_specified_nameSGD/m/dense_478/bias:62
0
_user_specified_nameSGD/m/dense_479/kernel:40
.
_user_specified_nameSGD/m/dense_479/bias:'#
!
_user_specified_name	total_1:' #
!
_user_specified_name	count_1:%!!

_user_specified_nametotal:%"!

_user_specified_namecount
�
�
*__inference_dense_477_layer_call_fn_555282

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_477_layer_call_and_return_conditional_losses_554950p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555276:&"
 
_user_specified_name555278
�

�
E__inference_dense_473_layer_call_and_return_conditional_losses_555213

inputs1
matmul_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_474_layer_call_and_return_conditional_losses_554902

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_478_layer_call_and_return_conditional_losses_554966

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_475_layer_call_and_return_conditional_losses_554918

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
.__inference_sequential_69_layer_call_fn_555094
dense_473_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_473_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_69_layer_call_and_return_conditional_losses_555028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:&"
 
_user_specified_name555064:&"
 
_user_specified_name555066:&"
 
_user_specified_name555068:&"
 
_user_specified_name555070:&"
 
_user_specified_name555072:&"
 
_user_specified_name555074:&"
 
_user_specified_name555076:&"
 
_user_specified_name555078:&	"
 
_user_specified_name555080:&
"
 
_user_specified_name555082:&"
 
_user_specified_name555084:&"
 
_user_specified_name555086:&"
 
_user_specified_name555088:&"
 
_user_specified_name555090
�
�
*__inference_dense_474_layer_call_fn_555222

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_474_layer_call_and_return_conditional_losses_554902p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555216:&"
 
_user_specified_name555218
�
�
.__inference_sequential_69_layer_call_fn_555061
dense_473_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_473_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_69_layer_call_and_return_conditional_losses_554989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:&"
 
_user_specified_name555031:&"
 
_user_specified_name555033:&"
 
_user_specified_name555035:&"
 
_user_specified_name555037:&"
 
_user_specified_name555039:&"
 
_user_specified_name555041:&"
 
_user_specified_name555043:&"
 
_user_specified_name555045:&	"
 
_user_specified_name555047:&
"
 
_user_specified_name555049:&"
 
_user_specified_name555051:&"
 
_user_specified_name555053:&"
 
_user_specified_name555055:&"
 
_user_specified_name555057
�S
�
!__inference__wrapped_model_554873
dense_473_inputI
6sequential_69_dense_473_matmul_readvariableop_resource:	�F
7sequential_69_dense_473_biasadd_readvariableop_resource:	�J
6sequential_69_dense_474_matmul_readvariableop_resource:
��F
7sequential_69_dense_474_biasadd_readvariableop_resource:	�J
6sequential_69_dense_475_matmul_readvariableop_resource:
��F
7sequential_69_dense_475_biasadd_readvariableop_resource:	�J
6sequential_69_dense_476_matmul_readvariableop_resource:
��F
7sequential_69_dense_476_biasadd_readvariableop_resource:	�J
6sequential_69_dense_477_matmul_readvariableop_resource:
��F
7sequential_69_dense_477_biasadd_readvariableop_resource:	�J
6sequential_69_dense_478_matmul_readvariableop_resource:
��F
7sequential_69_dense_478_biasadd_readvariableop_resource:	�I
6sequential_69_dense_479_matmul_readvariableop_resource:	�E
7sequential_69_dense_479_biasadd_readvariableop_resource:
identity��.sequential_69/dense_473/BiasAdd/ReadVariableOp�-sequential_69/dense_473/MatMul/ReadVariableOp�.sequential_69/dense_474/BiasAdd/ReadVariableOp�-sequential_69/dense_474/MatMul/ReadVariableOp�.sequential_69/dense_475/BiasAdd/ReadVariableOp�-sequential_69/dense_475/MatMul/ReadVariableOp�.sequential_69/dense_476/BiasAdd/ReadVariableOp�-sequential_69/dense_476/MatMul/ReadVariableOp�.sequential_69/dense_477/BiasAdd/ReadVariableOp�-sequential_69/dense_477/MatMul/ReadVariableOp�.sequential_69/dense_478/BiasAdd/ReadVariableOp�-sequential_69/dense_478/MatMul/ReadVariableOp�.sequential_69/dense_479/BiasAdd/ReadVariableOp�-sequential_69/dense_479/MatMul/ReadVariableOp�
-sequential_69/dense_473/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_473_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_69/dense_473/MatMulMatMuldense_473_input5sequential_69/dense_473/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_473/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_473_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_473/BiasAddBiasAdd(sequential_69/dense_473/MatMul:product:06sequential_69/dense_473/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_473/ReluRelu(sequential_69/dense_473/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_474/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_474_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_69/dense_474/MatMulMatMul*sequential_69/dense_473/Relu:activations:05sequential_69/dense_474/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_474/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_474_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_474/BiasAddBiasAdd(sequential_69/dense_474/MatMul:product:06sequential_69/dense_474/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_474/ReluRelu(sequential_69/dense_474/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_475/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_475_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_69/dense_475/MatMulMatMul*sequential_69/dense_474/Relu:activations:05sequential_69/dense_475/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_475/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_475_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_475/BiasAddBiasAdd(sequential_69/dense_475/MatMul:product:06sequential_69/dense_475/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_475/ReluRelu(sequential_69/dense_475/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_476/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_476_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_69/dense_476/MatMulMatMul*sequential_69/dense_475/Relu:activations:05sequential_69/dense_476/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_476/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_476_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_476/BiasAddBiasAdd(sequential_69/dense_476/MatMul:product:06sequential_69/dense_476/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_476/ReluRelu(sequential_69/dense_476/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_477/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_477_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_69/dense_477/MatMulMatMul*sequential_69/dense_476/Relu:activations:05sequential_69/dense_477/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_477/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_477_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_477/BiasAddBiasAdd(sequential_69/dense_477/MatMul:product:06sequential_69/dense_477/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_477/ReluRelu(sequential_69/dense_477/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_478/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_478_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_69/dense_478/MatMulMatMul*sequential_69/dense_477/Relu:activations:05sequential_69/dense_478/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.sequential_69/dense_478/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_478_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_69/dense_478/BiasAddBiasAdd(sequential_69/dense_478/MatMul:product:06sequential_69/dense_478/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
sequential_69/dense_478/ReluRelu(sequential_69/dense_478/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-sequential_69/dense_479/MatMul/ReadVariableOpReadVariableOp6sequential_69_dense_479_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_69/dense_479/MatMulMatMul*sequential_69/dense_478/Relu:activations:05sequential_69/dense_479/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.sequential_69/dense_479/BiasAdd/ReadVariableOpReadVariableOp7sequential_69_dense_479_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_69/dense_479/BiasAddBiasAdd(sequential_69/dense_479/MatMul:product:06sequential_69/dense_479/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
sequential_69/dense_479/SigmoidSigmoid(sequential_69/dense_479/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#sequential_69/dense_479/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp/^sequential_69/dense_473/BiasAdd/ReadVariableOp.^sequential_69/dense_473/MatMul/ReadVariableOp/^sequential_69/dense_474/BiasAdd/ReadVariableOp.^sequential_69/dense_474/MatMul/ReadVariableOp/^sequential_69/dense_475/BiasAdd/ReadVariableOp.^sequential_69/dense_475/MatMul/ReadVariableOp/^sequential_69/dense_476/BiasAdd/ReadVariableOp.^sequential_69/dense_476/MatMul/ReadVariableOp/^sequential_69/dense_477/BiasAdd/ReadVariableOp.^sequential_69/dense_477/MatMul/ReadVariableOp/^sequential_69/dense_478/BiasAdd/ReadVariableOp.^sequential_69/dense_478/MatMul/ReadVariableOp/^sequential_69/dense_479/BiasAdd/ReadVariableOp.^sequential_69/dense_479/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 2`
.sequential_69/dense_473/BiasAdd/ReadVariableOp.sequential_69/dense_473/BiasAdd/ReadVariableOp2^
-sequential_69/dense_473/MatMul/ReadVariableOp-sequential_69/dense_473/MatMul/ReadVariableOp2`
.sequential_69/dense_474/BiasAdd/ReadVariableOp.sequential_69/dense_474/BiasAdd/ReadVariableOp2^
-sequential_69/dense_474/MatMul/ReadVariableOp-sequential_69/dense_474/MatMul/ReadVariableOp2`
.sequential_69/dense_475/BiasAdd/ReadVariableOp.sequential_69/dense_475/BiasAdd/ReadVariableOp2^
-sequential_69/dense_475/MatMul/ReadVariableOp-sequential_69/dense_475/MatMul/ReadVariableOp2`
.sequential_69/dense_476/BiasAdd/ReadVariableOp.sequential_69/dense_476/BiasAdd/ReadVariableOp2^
-sequential_69/dense_476/MatMul/ReadVariableOp-sequential_69/dense_476/MatMul/ReadVariableOp2`
.sequential_69/dense_477/BiasAdd/ReadVariableOp.sequential_69/dense_477/BiasAdd/ReadVariableOp2^
-sequential_69/dense_477/MatMul/ReadVariableOp-sequential_69/dense_477/MatMul/ReadVariableOp2`
.sequential_69/dense_478/BiasAdd/ReadVariableOp.sequential_69/dense_478/BiasAdd/ReadVariableOp2^
-sequential_69/dense_478/MatMul/ReadVariableOp-sequential_69/dense_478/MatMul/ReadVariableOp2`
.sequential_69/dense_479/BiasAdd/ReadVariableOp.sequential_69/dense_479/BiasAdd/ReadVariableOp2^
-sequential_69/dense_479/MatMul/ReadVariableOp-sequential_69/dense_479/MatMul/ReadVariableOp:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
*__inference_dense_476_layer_call_fn_555262

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_476_layer_call_and_return_conditional_losses_554934p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555256:&"
 
_user_specified_name555258
�
�
*__inference_dense_475_layer_call_fn_555242

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_475_layer_call_and_return_conditional_losses_554918p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name555236:&"
 
_user_specified_name555238
�

�
E__inference_dense_479_layer_call_and_return_conditional_losses_555333

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_479_layer_call_and_return_conditional_losses_554982

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_475_layer_call_and_return_conditional_losses_555253

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
__inference__traced_save_555559
file_prefix:
'read_disablecopyonread_dense_473_kernel:	�6
'read_1_disablecopyonread_dense_473_bias:	�=
)read_2_disablecopyonread_dense_474_kernel:
��6
'read_3_disablecopyonread_dense_474_bias:	�=
)read_4_disablecopyonread_dense_475_kernel:
��6
'read_5_disablecopyonread_dense_475_bias:	�=
)read_6_disablecopyonread_dense_476_kernel:
��6
'read_7_disablecopyonread_dense_476_bias:	�=
)read_8_disablecopyonread_dense_477_kernel:
��6
'read_9_disablecopyonread_dense_477_bias:	�>
*read_10_disablecopyonread_dense_478_kernel:
��7
(read_11_disablecopyonread_dense_478_bias:	�=
*read_12_disablecopyonread_dense_479_kernel:	�6
(read_13_disablecopyonread_dense_479_bias:-
#read_14_disablecopyonread_iteration:	 1
'read_15_disablecopyonread_learning_rate: C
0read_16_disablecopyonread_sgd_m_dense_473_kernel:	�=
.read_17_disablecopyonread_sgd_m_dense_473_bias:	�D
0read_18_disablecopyonread_sgd_m_dense_474_kernel:
��=
.read_19_disablecopyonread_sgd_m_dense_474_bias:	�D
0read_20_disablecopyonread_sgd_m_dense_475_kernel:
��=
.read_21_disablecopyonread_sgd_m_dense_475_bias:	�D
0read_22_disablecopyonread_sgd_m_dense_476_kernel:
��=
.read_23_disablecopyonread_sgd_m_dense_476_bias:	�D
0read_24_disablecopyonread_sgd_m_dense_477_kernel:
��=
.read_25_disablecopyonread_sgd_m_dense_477_bias:	�D
0read_26_disablecopyonread_sgd_m_dense_478_kernel:
��=
.read_27_disablecopyonread_sgd_m_dense_478_bias:	�C
0read_28_disablecopyonread_sgd_m_dense_479_kernel:	�<
.read_29_disablecopyonread_sgd_m_dense_479_bias:+
!read_30_disablecopyonread_total_1: +
!read_31_disablecopyonread_count_1: )
read_32_disablecopyonread_total: )
read_33_disablecopyonread_count: 
savev2_const
identity_69��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_dense_473_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_dense_473_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_473_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_473_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_dense_474_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_dense_474_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_dense_474_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_dense_474_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_dense_475_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_dense_475_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_475_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_475_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_dense_476_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_dense_476_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_dense_476_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_dense_476_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_dense_477_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_dense_477_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_dense_477_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_dense_477_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_dense_478_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_dense_478_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_478_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_478_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:�
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_dense_479_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_dense_479_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	�}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_dense_479_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_dense_479_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_14/DisableCopyOnReadDisableCopyOnRead#read_14_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp#read_14_disablecopyonread_iteration^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_learning_rate^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_sgd_m_dense_473_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_sgd_m_dense_473_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnRead.read_17_disablecopyonread_sgd_m_dense_473_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp.read_17_disablecopyonread_sgd_m_dense_473_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead0read_18_disablecopyonread_sgd_m_dense_474_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp0read_18_disablecopyonread_sgd_m_dense_474_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_19/DisableCopyOnReadDisableCopyOnRead.read_19_disablecopyonread_sgd_m_dense_474_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp.read_19_disablecopyonread_sgd_m_dense_474_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_sgd_m_dense_475_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_sgd_m_dense_475_kernel^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_21/DisableCopyOnReadDisableCopyOnRead.read_21_disablecopyonread_sgd_m_dense_475_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp.read_21_disablecopyonread_sgd_m_dense_475_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_sgd_m_dense_476_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_sgd_m_dense_476_kernel^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_23/DisableCopyOnReadDisableCopyOnRead.read_23_disablecopyonread_sgd_m_dense_476_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp.read_23_disablecopyonread_sgd_m_dense_476_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead0read_24_disablecopyonread_sgd_m_dense_477_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp0read_24_disablecopyonread_sgd_m_dense_477_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_25/DisableCopyOnReadDisableCopyOnRead.read_25_disablecopyonread_sgd_m_dense_477_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp.read_25_disablecopyonread_sgd_m_dense_477_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_sgd_m_dense_478_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_sgd_m_dense_478_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_sgd_m_dense_478_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_sgd_m_dense_478_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead0read_28_disablecopyonread_sgd_m_dense_479_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp0read_28_disablecopyonread_sgd_m_dense_479_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_29/DisableCopyOnReadDisableCopyOnRead.read_29_disablecopyonread_sgd_m_dense_479_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp.read_29_disablecopyonread_sgd_m_dense_479_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_30/DisableCopyOnReadDisableCopyOnRead!read_30_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp!read_30_disablecopyonread_total_1^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_31/DisableCopyOnReadDisableCopyOnRead!read_31_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp!read_31_disablecopyonread_count_1^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_total^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_33/DisableCopyOnReadDisableCopyOnReadread_33_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOpread_33_disablecopyonread_count^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*�
value�B�#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *1
dtypes'
%2#	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_68Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_69IdentityIdentity_68:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_69Identity_69:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_user_specified_namedense_473/kernel:.*
(
_user_specified_namedense_473/bias:0,
*
_user_specified_namedense_474/kernel:.*
(
_user_specified_namedense_474/bias:0,
*
_user_specified_namedense_475/kernel:.*
(
_user_specified_namedense_475/bias:0,
*
_user_specified_namedense_476/kernel:.*
(
_user_specified_namedense_476/bias:0	,
*
_user_specified_namedense_477/kernel:.
*
(
_user_specified_namedense_477/bias:0,
*
_user_specified_namedense_478/kernel:.*
(
_user_specified_namedense_478/bias:0,
*
_user_specified_namedense_479/kernel:.*
(
_user_specified_namedense_479/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:62
0
_user_specified_nameSGD/m/dense_473/kernel:40
.
_user_specified_nameSGD/m/dense_473/bias:62
0
_user_specified_nameSGD/m/dense_474/kernel:40
.
_user_specified_nameSGD/m/dense_474/bias:62
0
_user_specified_nameSGD/m/dense_475/kernel:40
.
_user_specified_nameSGD/m/dense_475/bias:62
0
_user_specified_nameSGD/m/dense_476/kernel:40
.
_user_specified_nameSGD/m/dense_476/bias:62
0
_user_specified_nameSGD/m/dense_477/kernel:40
.
_user_specified_nameSGD/m/dense_477/bias:62
0
_user_specified_nameSGD/m/dense_478/kernel:40
.
_user_specified_nameSGD/m/dense_478/bias:62
0
_user_specified_nameSGD/m/dense_479/kernel:40
.
_user_specified_nameSGD/m/dense_479/bias:'#
!
_user_specified_name	total_1:' #
!
_user_specified_name	count_1:%!!

_user_specified_nametotal:%"!

_user_specified_namecount:=#9

_output_shapes
: 

_user_specified_nameConst
�

�
E__inference_dense_476_layer_call_and_return_conditional_losses_554934

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
E__inference_dense_476_layer_call_and_return_conditional_losses_555273

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
$__inference_signature_wrapper_555193
dense_473_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:
��
	unknown_8:	�
	unknown_9:
��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_473_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_554873o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:���������
)
_user_specified_namedense_473_input:&"
 
_user_specified_name555163:&"
 
_user_specified_name555165:&"
 
_user_specified_name555167:&"
 
_user_specified_name555169:&"
 
_user_specified_name555171:&"
 
_user_specified_name555173:&"
 
_user_specified_name555175:&"
 
_user_specified_name555177:&	"
 
_user_specified_name555179:&
"
 
_user_specified_name555181:&"
 
_user_specified_name555183:&"
 
_user_specified_name555185:&"
 
_user_specified_name555187:&"
 
_user_specified_name555189"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
K
dense_473_input8
!serving_default_dense_473_input:0���������=
	dense_4790
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
�
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13"
trackable_list_wrapper
�
0
1
2
 3
'4
(5
/6
07
78
89
?10
@11
G12
H13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_0
Otrace_12�
.__inference_sequential_69_layer_call_fn_555061
.__inference_sequential_69_layer_call_fn_555094�
���
FullArgSpec)
args!�
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
annotations� *
 zNtrace_0zOtrace_1
�
Ptrace_0
Qtrace_12�
I__inference_sequential_69_layer_call_and_return_conditional_losses_554989
I__inference_sequential_69_layer_call_and_return_conditional_losses_555028�
���
FullArgSpec)
args!�
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
annotations� *
 zPtrace_0zQtrace_1
�B�
!__inference__wrapped_model_554873dense_473_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
R
_variables
S_iterations
T_learning_rate
U_index_dict
V	momentums
W_update_step_xla"
experimentalOptimizer
,
Xserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
^trace_02�
*__inference_dense_473_layer_call_fn_555202�
���
FullArgSpec
args�

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
annotations� *
 z^trace_0
�
_trace_02�
E__inference_dense_473_layer_call_and_return_conditional_losses_555213�
���
FullArgSpec
args�

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
annotations� *
 z_trace_0
#:!	�2dense_473/kernel
:�2dense_473/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
etrace_02�
*__inference_dense_474_layer_call_fn_555222�
���
FullArgSpec
args�

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
annotations� *
 zetrace_0
�
ftrace_02�
E__inference_dense_474_layer_call_and_return_conditional_losses_555233�
���
FullArgSpec
args�

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
annotations� *
 zftrace_0
$:"
��2dense_474/kernel
:�2dense_474/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
ltrace_02�
*__inference_dense_475_layer_call_fn_555242�
���
FullArgSpec
args�

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
annotations� *
 zltrace_0
�
mtrace_02�
E__inference_dense_475_layer_call_and_return_conditional_losses_555253�
���
FullArgSpec
args�

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
annotations� *
 zmtrace_0
$:"
��2dense_475/kernel
:�2dense_475/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
strace_02�
*__inference_dense_476_layer_call_fn_555262�
���
FullArgSpec
args�

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
annotations� *
 zstrace_0
�
ttrace_02�
E__inference_dense_476_layer_call_and_return_conditional_losses_555273�
���
FullArgSpec
args�

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
annotations� *
 zttrace_0
$:"
��2dense_476/kernel
:�2dense_476/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
ztrace_02�
*__inference_dense_477_layer_call_fn_555282�
���
FullArgSpec
args�

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
annotations� *
 zztrace_0
�
{trace_02�
E__inference_dense_477_layer_call_and_return_conditional_losses_555293�
���
FullArgSpec
args�

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
annotations� *
 z{trace_0
$:"
��2dense_477/kernel
:�2dense_477/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_478_layer_call_fn_555302�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_478_layer_call_and_return_conditional_losses_555313�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
$:"
��2dense_478/kernel
:�2dense_478/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_dense_479_layer_call_fn_555322�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
�
�trace_02�
E__inference_dense_479_layer_call_and_return_conditional_losses_555333�
���
FullArgSpec
args�

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
annotations� *
 z�trace_0
#:!	�2dense_479/kernel
:2dense_479/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_69_layer_call_fn_555061dense_473_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_69_layer_call_fn_555094dense_473_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_69_layer_call_and_return_conditional_losses_554989dense_473_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_69_layer_call_and_return_conditional_losses_555028dense_473_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
S0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_555193dense_473_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 $

kwonlyargs�
jdense_473_input
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
*__inference_dense_473_layer_call_fn_555202inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_473_layer_call_and_return_conditional_losses_555213inputs"�
���
FullArgSpec
args�

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
*__inference_dense_474_layer_call_fn_555222inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_474_layer_call_and_return_conditional_losses_555233inputs"�
���
FullArgSpec
args�

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
*__inference_dense_475_layer_call_fn_555242inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_475_layer_call_and_return_conditional_losses_555253inputs"�
���
FullArgSpec
args�

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
*__inference_dense_476_layer_call_fn_555262inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_476_layer_call_and_return_conditional_losses_555273inputs"�
���
FullArgSpec
args�

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
*__inference_dense_477_layer_call_fn_555282inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_477_layer_call_and_return_conditional_losses_555293inputs"�
���
FullArgSpec
args�

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
*__inference_dense_478_layer_call_fn_555302inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_478_layer_call_and_return_conditional_losses_555313inputs"�
���
FullArgSpec
args�

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
*__inference_dense_479_layer_call_fn_555322inputs"�
���
FullArgSpec
args�

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
annotations� *
 
�B�
E__inference_dense_479_layer_call_and_return_conditional_losses_555333inputs"�
���
FullArgSpec
args�

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
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
':%	�2SGD/m/dense_473/kernel
!:�2SGD/m/dense_473/bias
(:&
��2SGD/m/dense_474/kernel
!:�2SGD/m/dense_474/bias
(:&
��2SGD/m/dense_475/kernel
!:�2SGD/m/dense_475/bias
(:&
��2SGD/m/dense_476/kernel
!:�2SGD/m/dense_476/bias
(:&
��2SGD/m/dense_477/kernel
!:�2SGD/m/dense_477/bias
(:&
��2SGD/m/dense_478/kernel
!:�2SGD/m/dense_478/bias
':%	�2SGD/m/dense_479/kernel
 :2SGD/m/dense_479/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_554873� '(/078?@GH8�5
.�+
)�&
dense_473_input���������
� "5�2
0
	dense_479#� 
	dense_479����������
E__inference_dense_473_layer_call_and_return_conditional_losses_555213d/�,
%�"
 �
inputs���������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_473_layer_call_fn_555202Y/�,
%�"
 �
inputs���������
� ""�
unknown�����������
E__inference_dense_474_layer_call_and_return_conditional_losses_555233e 0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_474_layer_call_fn_555222Z 0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_475_layer_call_and_return_conditional_losses_555253e'(0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_475_layer_call_fn_555242Z'(0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_476_layer_call_and_return_conditional_losses_555273e/00�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_476_layer_call_fn_555262Z/00�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_477_layer_call_and_return_conditional_losses_555293e780�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_477_layer_call_fn_555282Z780�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_478_layer_call_and_return_conditional_losses_555313e?@0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_dense_478_layer_call_fn_555302Z?@0�-
&�#
!�
inputs����������
� ""�
unknown�����������
E__inference_dense_479_layer_call_and_return_conditional_losses_555333dGH0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
*__inference_dense_479_layer_call_fn_555322YGH0�-
&�#
!�
inputs����������
� "!�
unknown����������
I__inference_sequential_69_layer_call_and_return_conditional_losses_554989� '(/078?@GH@�=
6�3
)�&
dense_473_input���������
p

 
� ",�)
"�
tensor_0���������
� �
I__inference_sequential_69_layer_call_and_return_conditional_losses_555028� '(/078?@GH@�=
6�3
)�&
dense_473_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
.__inference_sequential_69_layer_call_fn_555061u '(/078?@GH@�=
6�3
)�&
dense_473_input���������
p

 
� "!�
unknown����������
.__inference_sequential_69_layer_call_fn_555094u '(/078?@GH@�=
6�3
)�&
dense_473_input���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_555193� '(/078?@GHK�H
� 
A�>
<
dense_473_input)�&
dense_473_input���������"5�2
0
	dense_479#� 
	dense_479���������