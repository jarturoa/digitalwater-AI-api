▄Є/
Щ¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8ощ.
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:F*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
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
s
lstm/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 Ш*
shared_namelstm/kernel
l
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
_output_shapes
:	 Ш*
dtype0
З
lstm/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	FШ*&
shared_namelstm/recurrent_kernel
А
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
_output_shapes
:	FШ*
dtype0
k
	lstm/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*
shared_name	lstm/bias
d
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
_output_shapes	
:Ш*
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:F*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
Б
Adam/lstm/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 Ш*#
shared_nameAdam/lstm/kernel/m
z
&Adam/lstm/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/m*
_output_shapes
:	 Ш*
dtype0
Х
Adam/lstm/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	FШ*-
shared_nameAdam/lstm/recurrent_kernel/m
О
0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/m*
_output_shapes
:	FШ*
dtype0
y
Adam/lstm/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*!
shared_nameAdam/lstm/bias/m
r
$Adam/lstm/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/m*
_output_shapes	
:Ш*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:F*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
Б
Adam/lstm/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 Ш*#
shared_nameAdam/lstm/kernel/v
z
&Adam/lstm/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/kernel/v*
_output_shapes
:	 Ш*
dtype0
Х
Adam/lstm/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	FШ*-
shared_nameAdam/lstm/recurrent_kernel/v
О
0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/recurrent_kernel/v*
_output_shapes
:	FШ*
dtype0
y
Adam/lstm/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*!
shared_nameAdam/lstm/bias/v
r
$Adam/lstm/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/bias/v*
_output_shapes	
:Ш*
dtype0

NoOpNoOp
Щ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╘
value╩B╟ B└
╠
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
l

cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Ъ
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;
 
#
0
1
2
3
4
#
0
1
2
3
4
Ъ
layer_regularization_losses
regularization_losses

layers
 non_trainable_variables
trainable_variables
	variables
!metrics
 
~

kernel
recurrent_kernel
bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
 
 

0
1
2

0
1
2
Ъ
&layer_regularization_losses
regularization_losses

'layers
(non_trainable_variables
trainable_variables
	variables
)metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
*layer_regularization_losses
regularization_losses

+layers
,non_trainable_variables
trainable_variables
	variables
-metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUE	lstm/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
 

0
1
2

0
1
2
Ъ
.layer_regularization_losses
"regularization_losses

/layers
0non_trainable_variables
#trainable_variables
$	variables
1metrics
 


0
 
 
 
 
 
 
 
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/lstm/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/lstm/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Е
serving_default_lstm_inputPlaceholder*+
_output_shapes
:          *
dtype0* 
shape:          
ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/kernellstm/recurrent_kernel	lstm/biasdense/kernel
dense/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_225464
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
П
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm/bias/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp&Adam/lstm/kernel/m/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/m/Read/ReadVariableOp$Adam/lstm/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp&Adam/lstm/kernel/v/Read/ReadVariableOp0Adam/lstm/recurrent_kernel/v/Read/ReadVariableOp$Adam/lstm/bias/v/Read/ReadVariableOpConst*!
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_228287
·
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/kernellstm/recurrent_kernel	lstm/biasAdam/dense/kernel/mAdam/dense/bias/mAdam/lstm/kernel/mAdam/lstm/recurrent_kernel/mAdam/lstm/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/lstm/kernel/vAdam/lstm/recurrent_kernel/vAdam/lstm/bias/v* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_228359ыФ.
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_227090_227266
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_b374c17b-2c58-40f8-9edf-3c36bfe12e16*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2272652T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Х
∙
while_cond_226453
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_226453___redundant_placeholder0.
*while_cond_226453___redundant_placeholder1.
*while_cond_226453___redundant_placeholder2.
*while_cond_226453___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
╖
╩
%__inference_lstm_layer_call_fn_227276

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2249042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
рG
Ц
 __inference_standard_lstm_226986

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_226897*
condR
while_cond_226896*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_b374c17b-2c58-40f8-9edf-3c36bfe12e16*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
рG
Ц
 __inference_standard_lstm_225625

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_225536*
condR
while_cond_225535*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_baa4af42-a15c-42b8-bd11-378fe827cef3*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
К
х
@__inference_lstm_layer_call_and_return_conditional_losses_225347

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ъ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2250652
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╒
Ш
+__inference_sequential_layer_call_fn_226372

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2254162
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
В▄
∙
=__inference___backward_cudnn_lstm_with_fallback_223823_223999
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0Д
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeю
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :                  F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeФ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :                  F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like╢
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                   :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                   2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1▓
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                   2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*Ц
_input_shapesД
Б:         F:                  F:         F:         F: :                  F:::::         F:         F::                   :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_58aa5e71-61d1-457a-ae2d-d123ff0d3c98*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2239982T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ў-
я
while_body_225985
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
╬=
б
+__inference_cudnn_lstm_with_fallback_227548

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_a7a31e04-59b6-477c-8b11-f6fcd9096706*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
К
х
@__inference_lstm_layer_call_and_return_conditional_losses_224904

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ъ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2246222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╬=
б
+__inference_cudnn_lstm_with_fallback_225168

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_ae8eb5e8-b350-48d5-a7e0-cb64e2566441*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ў-
я
while_body_227799
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_224726_224902
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_f53e1cfc-3eba-4c25-93f0-292c37570ec2*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2249012T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
╬=
б
+__inference_cudnn_lstm_with_fallback_226177

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_4f7309a8-e7a0-473e-8512-274f9ca904e4*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЁK
╚
)__forward_cudnn_lstm_with_fallback_224901

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_f53e1cfc-3eba-4c25-93f0-292c37570ec2*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_224726_2249022
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
╬=
б
+__inference_cudnn_lstm_with_fallback_227089

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_b374c17b-2c58-40f8-9edf-3c36bfe12e16*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
╬=
б
+__inference_cudnn_lstm_with_fallback_225728

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_baa4af42-a15c-42b8-bd11-378fe827cef3*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
В▄
∙
=__inference___backward_cudnn_lstm_with_fallback_227992_228168
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0Д
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeю
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :                  F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeФ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :                  F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like╢
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                   :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                   2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1▓
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                   2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*Ц
_input_shapesД
Б:         F:                  F:         F:         F: :                  F:::::         F:         F::                   :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_99502476-5f98-444f-90aa-907fd64df7a6*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2281672T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ў-
я
while_body_226897
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_222484_222660
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_53808473-ded8-4fbd-8ef5-56343c6e18be*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2226592T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_225169_225345
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_ae8eb5e8-b350-48d5-a7e0-cb64e2566441*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2253442T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ў-
я
while_body_227356
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_225729_225905
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_baa4af42-a15c-42b8-bd11-378fe827cef3*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2259042T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ў-
я
while_body_224976
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
Х
∙
while_cond_225984
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_225984___redundant_placeholder0.
*while_cond_225984___redundant_placeholder1.
*while_cond_225984___redundant_placeholder2.
*while_cond_225984___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
р%
С
F__inference_sequential_layer_call_and_return_conditional_losses_226362

inputs'
#lstm_statefulpartitionedcall_args_3'
#lstm_statefulpartitionedcall_args_4'
#lstm_statefulpartitionedcall_args_5(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвlstm/StatefulPartitionedCallN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstЙ

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         F2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstС
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2
lstm/zeros_1Н
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0#lstm_statefulpartitionedcall_args_3#lstm_statefulpartitionedcall_args_4#lstm_statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2260742
lstm/StatefulPartitionedCallЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype02
dense/MatMul/ReadVariableOpд
dense/MatMulMatMul%lstm/StatefulPartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd╞
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ЁK
╚
)__forward_cudnn_lstm_with_fallback_228167

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_99502476-5f98-444f-90aa-907fd64df7a6*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_227992_2281682
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
с
Ь
+__inference_sequential_layer_call_fn_225424

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2254162
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
┐
ў
F__inference_sequential_layer_call_and_return_conditional_losses_225402

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCall└
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2253472
lstm/StatefulPartitionedCall║
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2253782
dense/StatefulPartitionedCall╣
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
с
Ь
+__inference_sequential_layer_call_fn_225445

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2254372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
╒
Ш
+__inference_sequential_layer_call_fn_226382

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2254372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Х
∙
while_cond_227355
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_227355___redundant_placeholder0.
*while_cond_227355___redundant_placeholder1.
*while_cond_227355___redundant_placeholder2.
*while_cond_227355___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
ОH
Ц
 __inference_standard_lstm_227445

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_227356*
condR
while_cond_227355*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╟

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*e
_input_shapesT
R:                   :         F:         F:::*=
api_implements+)lstm_a7a31e04-59b6-477c-8b11-f6fcd9096706*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Х
∙
while_cond_224532
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_224532___redundant_placeholder0.
*while_cond_224532___redundant_placeholder1.
*while_cond_224532___redundant_placeholder2.
*while_cond_224532___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
│
є
F__inference_sequential_layer_call_and_return_conditional_losses_225416

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCall╝
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2249042
lstm/StatefulPartitionedCall║
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2253782
dense/StatefulPartitionedCall╣
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
╡
Х
$__inference_signature_wrapper_225464

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_2226682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
ў-
я
while_body_226454
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_226647_226823
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_6e1e60ed-65ef-4ede-9b38-db569fbea433*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2268222T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
ЁK
╚
)__forward_cudnn_lstm_with_fallback_227724

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_a7a31e04-59b6-477c-8b11-f6fcd9096706*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_227549_2277252
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ь
х
@__inference_lstm_layer_call_and_return_conditional_losses_224001

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1є
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*o
_output_shapes]
[:         F:                  F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2237192
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ОH
Ц
 __inference_standard_lstm_224170

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_224081*
condR
while_cond_224080*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╟

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*e
_input_shapesT
R:                   :         F:         F:::*=
api_implements+)lstm_67e06152-1664-4bb6-8e8d-42479fb923e7*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
р%
С
F__inference_sequential_layer_call_and_return_conditional_losses_225913

inputs'
#lstm_statefulpartitionedcall_args_3'
#lstm_statefulpartitionedcall_args_4'
#lstm_statefulpartitionedcall_args_5(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвlstm/StatefulPartitionedCallN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stackВ
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1В
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2А
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slicef
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros/mul/yА
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessl
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros/packed/1Ч
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/ConstЙ

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         F2

lstm/zerosj
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros_1/mul/yЖ
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm/zeros_1/Less/yГ
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessp
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
lstm/zeros_1/packed/1Э
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/ConstС
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2
lstm/zeros_1Н
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0#lstm_statefulpartitionedcall_args_3#lstm_statefulpartitionedcall_args_4#lstm_statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2256252
lstm/StatefulPartitionedCallЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype02
dense/MatMul/ReadVariableOpд
dense/MatMulMatMul%lstm/StatefulPartitionedCall:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd╞
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ў-
я
while_body_222291
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
┐
ў
F__inference_sequential_layer_call_and_return_conditional_losses_225391

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCall└
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2249042
lstm/StatefulPartitionedCall║
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2253782
dense/StatefulPartitionedCall╣
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
│
є
F__inference_sequential_layer_call_and_return_conditional_losses_225437

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallвlstm/StatefulPartitionedCall╝
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2253472
lstm/StatefulPartitionedCall║
dense/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2253782
dense/StatefulPartitionedCall╣
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ЁK
╚
)__forward_cudnn_lstm_with_fallback_227265

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_b374c17b-2c58-40f8-9edf-3c36bfe12e16*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_227090_2272662
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
╬=
б
+__inference_cudnn_lstm_with_fallback_222483

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_53808473-ded8-4fbd-8ef5-56343c6e18be*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ОH
Ц
 __inference_standard_lstm_223719

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_223630*
condR
while_cond_223629*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╟

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*e
_input_shapesT
R:                   :         F:         F:::*=
api_implements+)lstm_58aa5e71-61d1-457a-ae2d-d123ff0d3c98*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЁK
╚
)__forward_cudnn_lstm_with_fallback_223998

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_58aa5e71-61d1-457a-ae2d-d123ff0d3c98*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_223823_2239992
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
рG
Ц
 __inference_standard_lstm_225065

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_224976*
condR
while_cond_224975*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_ae8eb5e8-b350-48d5-a7e0-cb64e2566441*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ц
┌
A__inference_dense_layer_call_and_return_conditional_losses_225378

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
К
х
@__inference_lstm_layer_call_and_return_conditional_losses_227268

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ъ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2269862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
┤V
╔

"__inference__traced_restore_228359
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate"
assignvariableop_7_lstm_kernel,
(assignvariableop_8_lstm_recurrent_kernel 
assignvariableop_9_lstm_bias+
'assignvariableop_10_adam_dense_kernel_m)
%assignvariableop_11_adam_dense_bias_m*
&assignvariableop_12_adam_lstm_kernel_m4
0assignvariableop_13_adam_lstm_recurrent_kernel_m(
$assignvariableop_14_adam_lstm_bias_m+
'assignvariableop_15_adam_dense_kernel_v)
%assignvariableop_16_adam_dense_bias_v*
&assignvariableop_17_adam_lstm_kernel_v4
0assignvariableop_18_adam_lstm_recurrent_kernel_v(
$assignvariableop_19_adam_lstm_bias_v
identity_21ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1Т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю

valueФ
BС
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╢
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesП
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityН
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1У
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2Т
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Ф
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ф
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5У
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ы
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Ф
AssignVariableOp_7AssignVariableOpassignvariableop_7_lstm_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ю
AssignVariableOp_8AssignVariableOp(assignvariableop_8_lstm_recurrent_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Т
AssignVariableOp_9AssignVariableOpassignvariableop_9_lstm_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10а
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_kernel_mIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Ю
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_dense_bias_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12Я
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_lstm_kernel_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13й
AssignVariableOp_13AssignVariableOp0assignvariableop_13_adam_lstm_recurrent_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Э
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_lstm_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15а
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Ю
AssignVariableOp_16AssignVariableOp%assignvariableop_16_adam_dense_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Я
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_lstm_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18й
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_lstm_recurrent_kernel_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19Э
AssignVariableOp_19AssignVariableOp$assignvariableop_19_adam_lstm_bias_vIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20г
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
╬=
б
+__inference_cudnn_lstm_with_fallback_224725

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_f53e1cfc-3eba-4c25-93f0-292c37570ec2*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Х
∙
while_cond_223629
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_223629___redundant_placeholder0.
*while_cond_223629___redundant_placeholder1.
*while_cond_223629___redundant_placeholder2.
*while_cond_223629___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
рG
Ц
 __inference_standard_lstm_222380

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_222291*
condR
while_cond_222290*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_53808473-ded8-4fbd-8ef5-56343c6e18be*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
╖
╩
%__inference_lstm_layer_call_fn_227284

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2253472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ў-
я
while_body_224533
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
рG
Ц
 __inference_standard_lstm_226543

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_226454*
condR
while_cond_226453*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_6e1e60ed-65ef-4ede-9b38-db569fbea433*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ц
┌
A__inference_dense_layer_call_and_return_conditional_losses_228196

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         F::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
д
ч
@__inference_lstm_layer_call_and_return_conditional_losses_227727
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ї
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*o
_output_shapes]
[:         F:                  F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2274452
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
╞
╠
%__inference_lstm_layer_call_fn_228178
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2240012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
и1
╘
__inference__traced_save_228287
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop*
&savev2_lstm_kernel_read_readvariableop4
0savev2_lstm_recurrent_kernel_read_readvariableop(
$savev2_lstm_bias_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop1
-savev2_adam_lstm_kernel_m_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop/
+savev2_adam_lstm_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop1
-savev2_adam_lstm_kernel_v_read_readvariableop;
7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop/
+savev2_adam_lstm_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1е
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5e5a90ca56cd42acad6028c8f2575a8d/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameМ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ю

valueФ
BС
B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names░
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┴
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop&savev2_lstm_kernel_read_readvariableop0savev2_lstm_recurrent_kernel_read_readvariableop$savev2_lstm_bias_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop-savev2_adam_lstm_kernel_m_read_readvariableop7savev2_adam_lstm_recurrent_kernel_m_read_readvariableop+savev2_adam_lstm_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop-savev2_adam_lstm_kernel_v_read_readvariableop7savev2_adam_lstm_recurrent_kernel_v_read_readvariableop+savev2_adam_lstm_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*к
_input_shapesШ
Х: :F:: : : : : :	 Ш:	FШ:Ш:F::	 Ш:	FШ:Ш:F::	 Ш:	FШ:Ш: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ь
з
&__inference_dense_layer_call_fn_228203

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2253782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         F::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
В▄
∙
=__inference___backward_cudnn_lstm_with_fallback_227549_227725
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0Д
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeю
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :                  F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeФ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :                  F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like╢
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                   :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                   2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1▓
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                   2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*Ц
_input_shapesД
Б:         F:                  F:         F:         F: :                  F:::::         F:         F::                   :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_a7a31e04-59b6-477c-8b11-f6fcd9096706*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2277242T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
╬=
б
+__inference_cudnn_lstm_with_fallback_223822

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_58aa5e71-61d1-457a-ae2d-d123ff0d3c98*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЁK
╚
)__forward_cudnn_lstm_with_fallback_226353

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_4f7309a8-e7a0-473e-8512-274f9ca904e4*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_226178_2263542
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЁK
╚
)__forward_cudnn_lstm_with_fallback_225344

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_ae8eb5e8-b350-48d5-a7e0-cb64e2566441*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_225169_2253452
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
╬=
б
+__inference_cudnn_lstm_with_fallback_227991

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_99502476-5f98-444f-90aa-907fd64df7a6*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
К
х
@__inference_lstm_layer_call_and_return_conditional_losses_226825

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ъ
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2265432
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╞
╠
%__inference_lstm_layer_call_fn_228186
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:         F**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_lstm_layer_call_and_return_conditional_losses_2244522
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
ЁK
╚
)__forward_cudnn_lstm_with_fallback_225904

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_baa4af42-a15c-42b8-bd11-378fe827cef3*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_225729_2259052
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Х
∙
while_cond_224080
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_224080___redundant_placeholder0.
*while_cond_224080___redundant_placeholder1.
*while_cond_224080___redundant_placeholder2.
*while_cond_224080___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
Х
∙
while_cond_227798
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_227798___redundant_placeholder0.
*while_cond_227798___redundant_placeholder1.
*while_cond_227798___redundant_placeholder2.
*while_cond_227798___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
ЁK
╚
)__forward_cudnn_lstm_with_fallback_224449

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_67e06152-1664-4bb6-8e8d-42479fb923e7*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_224274_2244502
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
рG
Ц
 __inference_standard_lstm_224622

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_224533*
condR
while_cond_224532*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_f53e1cfc-3eba-4c25-93f0-292c37570ec2*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЁK
╚
)__forward_cudnn_lstm_with_fallback_226822

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_6e1e60ed-65ef-4ede-9b38-db569fbea433*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_226647_2268232
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ў-
я
while_body_223630
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
ЁK
╚
)__forward_cudnn_lstm_with_fallback_222659

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permO
transpose_0	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimR

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimX
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisР

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1{
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"3
strided_slice_stackstrided_slice/stack:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0*
_input_shapes *=
api_implements+)lstm_53808473-ded8-4fbd-8ef5-56343c6e18be*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_cudnn_lstm_with_fallback_222484_2226602
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ў-
я
while_body_225536
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
д
ч
@__inference_lstm_layer_call_and_return_conditional_losses_228170
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1ї
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*o
_output_shapes]
[:         F:                  F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2278882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
Х
∙
while_cond_222290
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_222290___redundant_placeholder0.
*while_cond_222290___redundant_placeholder1.
*while_cond_222290___redundant_placeholder2.
*while_cond_222290___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
Х
∙
while_cond_224975
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_224975___redundant_placeholder0.
*while_cond_224975___redundant_placeholder1.
*while_cond_224975___redundant_placeholder2.
*while_cond_224975___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
й.
╚
!__inference__wrapped_model_222668

lstm_input2
.sequential_lstm_statefulpartitionedcall_args_32
.sequential_lstm_statefulpartitionedcall_args_42
.sequential_lstm_statefulpartitionedcall_args_53
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identityИв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв'sequential/lstm/StatefulPartitionedCallh
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:2
sequential/lstm/ShapeФ
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stackШ
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1Ш
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2┬
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice|
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
sequential/lstm/zeros/mul/yм
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
sequential/lstm/zeros/Less/yз
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/LessВ
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2 
sequential/lstm/zeros/packed/1├
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/Const╡
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         F2
sequential/lstm/zerosА
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
sequential/lstm/zeros_1/mul/y▓
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mulГ
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2 
sequential/lstm/zeros_1/Less/yп
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/LessЖ
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2"
 sequential/lstm/zeros_1/packed/1╔
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packedГ
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/Const╜
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2
sequential/lstm/zeros_1▐
'sequential/lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputsequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0.sequential_lstm_statefulpartitionedcall_args_3.sequential_lstm_statefulpartitionedcall_args_4.sequential_lstm_statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*f
_output_shapesT
R:         F:         F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2223802)
'sequential/lstm/StatefulPartitionedCall└
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:F*
dtype02(
&sequential/dense/MatMul/ReadVariableOp╨
sequential/dense/MatMulMatMul0sequential/lstm/StatefulPartitionedCall:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/MatMul┐
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp┼
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential/dense/BiasAddЄ
IdentityIdentity!sequential/dense/BiasAdd:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp(^sequential/lstm/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:          :::::2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2R
'sequential/lstm/StatefulPartitionedCall'sequential/lstm/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input
рG
Ц
 __inference_standard_lstm_226074

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:          2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_225985*
condR
while_cond_225984*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╛

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*\
_input_shapesK
I:          :         F:         F:::*=
api_implements+)lstm_4f7309a8-e7a0-473e-8512-274f9ca904e4*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Х
∙
while_cond_225535
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_225535___redundant_placeholder0.
*while_cond_225535___redundant_placeholder1.
*while_cond_225535___redundant_placeholder2.
*while_cond_225535___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
╬=
б
+__inference_cudnn_lstm_with_fallback_226646

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_6e1e60ed-65ef-4ede-9b38-db569fbea433*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
з█
∙
=__inference___backward_cudnn_lstm_with_fallback_226178_226354
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0{
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeх
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*+
_output_shapes
:         F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutation▀
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeЛ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*+
_output_shapes
:         F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_likeн
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:          :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutationЎ
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:          2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1й
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:          2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*√
_input_shapesщ
ц:         F:         F:         F:         F: :         F:::::         F:         F::          :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_4f7309a8-e7a0-473e-8512-274f9ca904e4*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2263532T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop
Ь
х
@__inference_lstm_layer_call_and_return_conditional_losses_224452

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityИвStatefulPartitionedCallD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         F2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :F2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         F2	
zeros_1є
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*
Tout	
2*,
_gradient_op_typePartitionedCallUnused*o
_output_shapes]
[:         F:                  F:         F:         F: **
config_proto

CPU

GPU 2J 8*)
f$R"
 __inference_standard_lstm_2241702
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         F2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:                   :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Х
∙
while_cond_226896
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice.
*while_cond_226896___redundant_placeholder0.
*while_cond_226896___redundant_placeholder1.
*while_cond_226896___redundant_placeholder2.
*while_cond_226896___redundant_placeholder3
identity
V
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*S
_input_shapesB
@: : : : :         F:         F: ::::
ОH
Ц
 __inference_standard_lstm_227888

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвwhileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                   2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceЕ
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape░
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2№
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:          *
shrink_axis_mask2
strided_slice_1v
MatMul/ReadVariableOpReadVariableOpkernel*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpЖ
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulД
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addr
BiasAdd/ReadVariableOpReadVariableOpbias*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1Z
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   2
TensorArrayV2_1/element_shape╢
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterФ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         F:         F: : : : : *
bodyR
while_body_227799*
condR
while_cond_227798*K
output_shapes:
8: : : : :         F:         F: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    F   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  F*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2Ъ
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         F*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  F2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?2	
runtime┐
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity╟

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  F2

Identity_1╣

Identity_2Identitywhile:output:4^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_2╣

Identity_3Identitywhile:output:5^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         F2

Identity_3к

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*e
_input_shapesT
R:                   :         F:         F:::*=
api_implements+)lstm_99502476-5f98-444f-90aa-907fd64df7a6*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ў-
я
while_body_224081
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOp╖
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        23
1TensorArrayV2Read/TensorListGetItem/element_shape╡
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:          *
element_dtype02%
#TensorArrayV2Read/TensorListGetItemР
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0*
_output_shapes
:	 Ш*
dtype02
MatMul/ReadVariableOpШ
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulЦ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0*
_output_shapes
:	FШ*
dtype02
MatMul_1/ReadVariableOpБ
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addП
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         F:         F:         F:         F*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         F2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         F2
	Sigmoid_1a
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         F2
mulV
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         F2
Tanh^
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         F2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         F2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         F2
	Sigmoid_2U
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         F2
Tanh_1b
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         F2
mul_2╡
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yW
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: 2
add_2T
add_3/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_3/y^
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: 2
add_3Ч
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identityк

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1Ы

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_2╞

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_3м

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_4м

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         F2

Identity_5"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0" 
strided_slicestrided_slice_0"Ь
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :         F:         F: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp
╬=
б
+__inference_cudnn_lstm_with_fallback_224273

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ИвCudnnRNNвconcat/ReadVariableOpвsplit/ReadVariableOpвsplit_1/ReadVariableOpY
transpose/permConst*
dtype0*!
valueB"          2
transpose/permM
	transpose	Transposeinputstranspose/perm:output:0*
T02
	transposeJ
ExpandDims/dimConst*
dtype0*
value	B : 2
ExpandDims/dimP

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T02

ExpandDimsN
ExpandDims_1/dimConst*
dtype0*
value	B : 2
ExpandDims_1/dimV
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T02
ExpandDims_18
ConstConst*
dtype0*
value	B :2
ConstL
split/split_dimConst*
dtype0*
value	B :2
split/split_dimS
split/ReadVariableOpReadVariableOpkernel*
dtype02
split/ReadVariableOpi
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split2
split<
Const_1Const*
dtype0*
value	B :2	
Const_1P
split_1/split_dimConst*
dtype0*
value	B :2
split_1/split_dima
split_1/ReadVariableOpReadVariableOprecurrent_kernel*
dtype02
split_1/ReadVariableOpq
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_split2	
split_1J

zeros_likeConst*
dtype0*
valueBШ*    2

zeros_likeS
concat/ReadVariableOpReadVariableOpbias*
dtype02
concat/ReadVariableOpD
concat/axisConst*
dtype0*
value	B : 2
concat/axisx
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T02
concat<
Const_2Const*
dtype0*
value	B :2	
Const_2P
split_2/split_dimConst*
dtype0*
value	B : 2
split_2/split_dimb
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split2	
split_2I
Const_3Const*
dtype0*
valueB:
         2	
Const_3Y
transpose_1/permConst*
dtype0*
valueB"       2
transpose_1/perm[
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T02
transpose_1I
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T02	
ReshapeY
transpose_2/permConst*
dtype0*
valueB"       2
transpose_2/perm[
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T02
transpose_2M
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T02
	Reshape_1Y
transpose_3/permConst*
dtype0*
valueB"       2
transpose_3/perm[
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T02
transpose_3M
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T02
	Reshape_2Y
transpose_4/permConst*
dtype0*
valueB"       2
transpose_4/perm[
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T02
transpose_4M
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T02
	Reshape_3Y
transpose_5/permConst*
dtype0*
valueB"       2
transpose_5/perm]
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T02
transpose_5M
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T02
	Reshape_4Y
transpose_6/permConst*
dtype0*
valueB"       2
transpose_6/perm]
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T02
transpose_6M
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T02
	Reshape_5Y
transpose_7/permConst*
dtype0*
valueB"       2
transpose_7/perm]
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T02
transpose_7M
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T02
	Reshape_6Y
transpose_8/permConst*
dtype0*
valueB"       2
transpose_8/perm]
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T02
transpose_8M
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T02
	Reshape_7N
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T02
	Reshape_8N
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T02
	Reshape_9P

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T02

Reshape_10P

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T02

Reshape_11P

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T02

Reshape_12P

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T02

Reshape_13P

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T02

Reshape_14P

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T02

Reshape_15H
concat_1/axisConst*
dtype0*
value	B : 2
concat_1/axisО
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T02

concat_1w
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T02

CudnnRNNa
strided_slice/stackConst*
dtype0*
valueB:
         2
strided_slice/stack\
strided_slice/stack_1Const*
dtype0*
valueB: 2
strided_slice/stack_1\
strided_slice/stack_2Const*
dtype0*
valueB:2
strided_slice/stack_2═
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask2
strided_slice]
transpose_9/permConst*
dtype0*!
valueB"          2
transpose_9/perm^
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T02
transpose_9R
SqueezeSqueezeCudnnRNN:output_h:0*
T0*
squeeze_dims
 2	
SqueezeV
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*
squeeze_dims
 2
	Squeeze_1N
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @2	
runtimeФ
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

IdentityС

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_1Т

Identity_2IdentitySqueeze:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_2Ф

Identity_3IdentitySqueeze_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_3Т

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp*
T02

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_67e06152-1664-4bb6-8e8d-42479fb923e7*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
В▄
∙
=__inference___backward_cudnn_lstm_with_fallback_224274_224450
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5Ив(gradients/CudnnRNN_grad/CudnnRNNBackpropu
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         F2
gradients/grad_ys_0Д
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  F2
gradients/grad_ys_1w
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         F2
gradients/grad_ys_2w
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         F2
gradients/grad_ys_3f
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: 2
gradients/grad_ys_4г
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:2$
"gradients/strided_slice_grad/Shapeю
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :                  F*
shrink_axis_mask2/
-gradients/strided_slice_grad/StridedSliceGrad╠
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:2.
,gradients/transpose_9_grad/InvertPermutationш
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  F2&
$gradients/transpose_9_grad/transposeС
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:2
gradients/Squeeze_grad/Shape╞
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:         F2 
gradients/Squeeze_grad/ReshapeЧ
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:2 
gradients/Squeeze_1_grad/Shape╠
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:         F2"
 gradients/Squeeze_1_grad/ReshapeФ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :                  F2
gradients/AddN{
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:2
gradients/zeros_like╢
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                   :         F:         F:└у2*
(gradients/CudnnRNN_grad/CudnnRNNBackprop─
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:2,
*gradients/transpose_grad/InvertPermutation 
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                   2$
"gradients/transpose_grad/transposeШ
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:2!
gradients/ExpandDims_grad/Shapeъ
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         F2#
!gradients/ExpandDims_grad/ReshapeЮ
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:2#
!gradients/ExpandDims_1_grad/ShapeЁ
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         F2%
#gradients/ExpandDims_1_grad/Reshape~
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_1_grad/Rank╣
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_1_grad/modЙ
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:└2
gradients/concat_1_grad/ShapeН
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_1Н
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_2Н
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:└2!
gradients/concat_1_grad/Shape_3Н
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_4Н
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_5Н
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_6Н
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:д&2!
gradients/concat_1_grad/Shape_7М
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_8М
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/concat_1_grad/Shape_9О
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_10О
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_11О
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_12О
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_13О
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_14О
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB:F2"
 gradients/concat_1_grad/Shape_15а
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::2&
$gradients/concat_1_grad/ConcatOffsetН
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:└2
gradients/concat_1_grad/SliceУ
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_1У
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_2У
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:└2!
gradients/concat_1_grad/Slice_3У
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_4У
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_5У
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_6У
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:д&2!
gradients/concat_1_grad/Slice_7Т
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_8Т
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:F2!
gradients/concat_1_grad/Slice_9Ц
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_10Ц
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_11Ц
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_12Ц
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_13Ц
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_14Ц
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:F2"
 gradients/concat_1_grad/Slice_15Н
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2
gradients/Reshape_grad/Shape├
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:F 2 
gradients/Reshape_grad/ReshapeС
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_1_grad/Shape╦
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_1_grad/ReshapeС
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_2_grad/Shape╦
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_2_grad/ReshapeС
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F       2 
gradients/Reshape_3_grad/Shape╦
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:F 2"
 gradients/Reshape_3_grad/ReshapeС
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_4_grad/Shape╦
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_4_grad/ReshapeС
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_5_grad/Shape╦
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_5_grad/ReshapeС
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_6_grad/Shape╦
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_6_grad/ReshapeС
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"F   F   2 
gradients/Reshape_7_grad/Shape╦
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:FF2"
 gradients/Reshape_7_grad/ReshapeК
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_8_grad/Shape╟
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_8_grad/ReshapeК
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2 
gradients/Reshape_9_grad/Shape╟
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:F2"
 gradients/Reshape_9_grad/ReshapeМ
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_10_grad/Shape╦
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_10_grad/ReshapeМ
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_11_grad/Shape╦
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_11_grad/ReshapeМ
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_12_grad/Shape╦
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_12_grad/ReshapeМ
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_13_grad/Shape╦
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_13_grad/ReshapeМ
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_14_grad/Shape╦
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_14_grad/ReshapeМ
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:F2!
gradients/Reshape_15_grad/Shape╦
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:F2#
!gradients/Reshape_15_grad/Reshape╠
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:2.
,gradients/transpose_1_grad/InvertPermutation▌
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_1_grad/transpose╠
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:2.
,gradients/transpose_2_grad/InvertPermutation▀
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_2_grad/transpose╠
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:2.
,gradients/transpose_3_grad/InvertPermutation▀
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_3_grad/transpose╠
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:2.
,gradients/transpose_4_grad/InvertPermutation▀
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

: F2&
$gradients/transpose_4_grad/transpose╠
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:2.
,gradients/transpose_5_grad/InvertPermutation▀
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_5_grad/transpose╠
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:2.
,gradients/transpose_6_grad/InvertPermutation▀
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_6_grad/transpose╠
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:2.
,gradients/transpose_7_grad/InvertPermutation▀
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_7_grad/transpose╠
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:2.
,gradients/transpose_8_grad/InvertPermutation▀
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:FF2&
$gradients/transpose_8_grad/transposeИ
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:░2
gradients/split_2_grad/concat╬
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	 Ш2
gradients/split_grad/concat╓
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	FШ2
gradients/split_1_grad/concatz
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :2
gradients/concat_grad/Rankп
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: 2
gradients/concat_grad/modЕ
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/ShapeЙ
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:Ш2
gradients/concat_grad/Shape_1Ё
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::2$
"gradients/concat_grad/ConcatOffsetё
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Sliceў
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:Ш2
gradients/concat_grad/Slice_1▓
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                   2

Identityн

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_1п

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         F2

Identity_2Я

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	 Ш2

Identity_3б

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	FШ2

Identity_4Э

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:Ш2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*Ц
_input_shapesД
Б:         F:                  F:         F:         F: :                  F:::::         F:         F::                   :         F:         F:└у::         F:         F: ::::::::: : : : *=
api_implements+)lstm_67e06152-1664-4bb6-8e8d-42479fb923e7*
api_preferred_deviceGPU*D
forward_function_name+)__forward_cudnn_lstm_with_fallback_2244492T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop"пL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▓
serving_defaultЮ
E

lstm_input7
serving_default_lstm_input:0          9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict:жВ
╝
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
*<&call_and_return_all_conditional_losses
=__call__
>_default_save_signature"Ц
_tf_keras_sequentialў{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 1, 32], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 70, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 32], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 1, 32], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 70, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
л"и
_tf_keras_input_layerИ{"class_name": "InputLayer", "name": "lstm_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 1, 32], "config": {"batch_input_shape": [null, 1, 32], "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_input"}}
╣


cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"Р	
_tf_keras_layerЎ{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 1, 32], "config": {"name": "lstm", "trainable": true, "batch_input_shape": [null, 1, 32], "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 70, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 32], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
ю

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"╔
_tf_keras_layerп{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}}}
н
iter

beta_1

beta_2
	decay
learning_ratem2m3m4m5m6v7v8v9v:v;"
	optimizer
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
╖
layer_regularization_losses
regularization_losses

layers
 non_trainable_variables
trainable_variables
	variables
!metrics
=__call__
>_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
ю

kernel
recurrent_kernel
bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
*D&call_and_return_all_conditional_losses
E__call__"│
_tf_keras_layerЩ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 70, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
Ъ
&layer_regularization_losses
regularization_losses

'layers
(non_trainable_variables
trainable_variables
	variables
)metrics
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
:F2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ
*layer_regularization_losses
regularization_losses

+layers
,non_trainable_variables
trainable_variables
	variables
-metrics
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:	 Ш2lstm/kernel
(:&	FШ2lstm/recurrent_kernel
:Ш2	lstm/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
Ъ
.layer_regularization_losses
"regularization_losses

/layers
0non_trainable_variables
#trainable_variables
$	variables
1metrics
E__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!F2Adam/dense/kernel/m
:2Adam/dense/bias/m
#:!	 Ш2Adam/lstm/kernel/m
-:+	FШ2Adam/lstm/recurrent_kernel/m
:Ш2Adam/lstm/bias/m
#:!F2Adam/dense/kernel/v
:2Adam/dense/bias/v
#:!	 Ш2Adam/lstm/kernel/v
-:+	FШ2Adam/lstm/recurrent_kernel/v
:Ш2Adam/lstm/bias/v
ц2у
F__inference_sequential_layer_call_and_return_conditional_losses_226362
F__inference_sequential_layer_call_and_return_conditional_losses_225391
F__inference_sequential_layer_call_and_return_conditional_losses_225913
F__inference_sequential_layer_call_and_return_conditional_losses_225402└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·2ў
+__inference_sequential_layer_call_fn_226372
+__inference_sequential_layer_call_fn_225445
+__inference_sequential_layer_call_fn_225424
+__inference_sequential_layer_call_fn_226382└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
!__inference__wrapped_model_222668╜
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *-в*
(К%

lstm_input          
у2р
@__inference_lstm_layer_call_and_return_conditional_losses_227268
@__inference_lstm_layer_call_and_return_conditional_losses_228170
@__inference_lstm_layer_call_and_return_conditional_losses_227727
@__inference_lstm_layer_call_and_return_conditional_losses_226825╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ў2Ї
%__inference_lstm_layer_call_fn_228178
%__inference_lstm_layer_call_fn_227276
%__inference_lstm_layer_call_fn_227284
%__inference_lstm_layer_call_fn_228186╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_228196в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_228203в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
6B4
$__inference_signature_wrapper_225464
lstm_input
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 Ф
!__inference__wrapped_model_222668o7в4
-в*
(К%

lstm_input          
к "-к*
(
denseК
dense         б
A__inference_dense_layer_call_and_return_conditional_losses_228196\/в,
%в"
 К
inputs         F
к "%в"
К
0         
Ъ y
&__inference_dense_layer_call_fn_228203O/в,
%в"
 К
inputs         F
к "К         ▒
@__inference_lstm_layer_call_and_return_conditional_losses_226825m?в<
5в2
$К!
inputs          

 
p

 
к "%в"
К
0         F
Ъ ▒
@__inference_lstm_layer_call_and_return_conditional_losses_227268m?в<
5в2
$К!
inputs          

 
p 

 
к "%в"
К
0         F
Ъ ┴
@__inference_lstm_layer_call_and_return_conditional_losses_227727}OвL
EвB
4Ъ1
/К,
inputs/0                   

 
p

 
к "%в"
К
0         F
Ъ ┴
@__inference_lstm_layer_call_and_return_conditional_losses_228170}OвL
EвB
4Ъ1
/К,
inputs/0                   

 
p 

 
к "%в"
К
0         F
Ъ Й
%__inference_lstm_layer_call_fn_227276`?в<
5в2
$К!
inputs          

 
p

 
к "К         FЙ
%__inference_lstm_layer_call_fn_227284`?в<
5в2
$К!
inputs          

 
p 

 
к "К         FЩ
%__inference_lstm_layer_call_fn_228178pOвL
EвB
4Ъ1
/К,
inputs/0                   

 
p

 
к "К         FЩ
%__inference_lstm_layer_call_fn_228186pOвL
EвB
4Ъ1
/К,
inputs/0                   

 
p 

 
к "К         F╣
F__inference_sequential_layer_call_and_return_conditional_losses_225391o?в<
5в2
(К%

lstm_input          
p

 
к "%в"
К
0         
Ъ ╣
F__inference_sequential_layer_call_and_return_conditional_losses_225402o?в<
5в2
(К%

lstm_input          
p 

 
к "%в"
К
0         
Ъ ╡
F__inference_sequential_layer_call_and_return_conditional_losses_225913k;в8
1в.
$К!
inputs          
p

 
к "%в"
К
0         
Ъ ╡
F__inference_sequential_layer_call_and_return_conditional_losses_226362k;в8
1в.
$К!
inputs          
p 

 
к "%в"
К
0         
Ъ С
+__inference_sequential_layer_call_fn_225424b?в<
5в2
(К%

lstm_input          
p

 
к "К         С
+__inference_sequential_layer_call_fn_225445b?в<
5в2
(К%

lstm_input          
p 

 
к "К         Н
+__inference_sequential_layer_call_fn_226372^;в8
1в.
$К!
inputs          
p

 
к "К         Н
+__inference_sequential_layer_call_fn_226382^;в8
1в.
$К!
inputs          
p 

 
к "К         е
$__inference_signature_wrapper_225464}EвB
в 
;к8
6

lstm_input(К%

lstm_input          "-к*
(
denseК
dense         