--
-- TensorFlow C API FFI declarations and library loading.
--
local ffi = require("ffi")

ffi.cdef[[
typedef struct TF_Status TF_Status;
typedef struct TF_Graph TF_Graph;
typedef struct TF_Session TF_Session;
typedef struct TF_SessionOptions TF_SessionOptions;
typedef struct TF_Operation TF_Operation;
typedef struct TF_OperationDescription TF_OperationDescription;
typedef struct TF_Tensor TF_Tensor;
typedef struct TF_CheckpointReader TF_CheckpointReader;
typedef struct TFE_ContextOptions TFE_ContextOptions;
typedef struct TFE_Context TFE_Context;
typedef struct TFE_TensorHandle TFE_TensorHandle;
typedef struct TFE_Op TFE_Op;
typedef struct TF_TString { uint8_t raw[24]; } TF_TString;
typedef struct TF_ImportGraphDefOptions TF_ImportGraphDefOptions;
typedef struct TF_Buffer {
  const void* data;
  size_t length;
  void (*data_deallocator)(void* data, size_t length);
} TF_Buffer;
typedef struct TF_Output {
  TF_Operation* oper;
  int index;
} TF_Output;
typedef struct TF_Input {
  TF_Operation* oper;
  int index;
} TF_Input;
typedef enum TF_Code {
  TF_OK = 0, TF_CANCELLED = 1, TF_UNKNOWN = 2, TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4, TF_NOT_FOUND = 5, TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7, TF_RESOURCE_EXHAUSTED = 8, TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10, TF_OUT_OF_RANGE = 11, TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13, TF_UNAVAILABLE = 14, TF_DATA_LOSS = 15,
  TF_UNAUTHENTICATED = 16
} TF_Code;
typedef enum TF_DataType {
  TF_FLOAT = 1, TF_DOUBLE = 2, TF_INT32 = 3, TF_UINT8 = 4, TF_INT16 = 5,
  TF_INT8 = 6, TF_STRING = 7, TF_COMPLEX64 = 8, TF_INT64 = 9, TF_BOOL = 10,
  TF_QINT8 = 11, TF_QUINT8 = 12, TF_QINT32 = 13, TF_BFLOAT16 = 14,
  TF_QINT16 = 15, TF_QUINT16 = 16, TF_UINT16 = 17, TF_COMPLEX128 = 18,
  TF_HALF = 19, TF_RESOURCE = 20, TF_VARIANT = 21, TF_UINT32 = 22,
  TF_UINT64 = 23
} TF_DataType;
typedef enum TF_AttrType {
  TF_ATTR_STRING = 0, TF_ATTR_INT = 1, TF_ATTR_FLOAT = 2, TF_ATTR_BOOL = 3,
  TF_ATTR_TYPE = 4, TF_ATTR_SHAPE = 5, TF_ATTR_TENSOR = 6,
  TF_ATTR_PLACEHOLDER = 7, TF_ATTR_FUNC = 8
} TF_AttrType;
typedef struct TF_AttrMetadata {
  unsigned char is_list;
  int64_t list_size;
  TF_AttrType type;
  int64_t total_size;
} TF_AttrMetadata;

void free(void*);
const char* TF_Version(void);
TF_Status* TF_NewStatus(void);
void TF_DeleteStatus(TF_Status*);
TF_Code TF_GetCode(const TF_Status*);
const char* TF_Message(const TF_Status*);

TF_Graph* TF_NewGraph(void);
void TF_DeleteGraph(TF_Graph*);
TF_Operation* TF_GraphOperationByName(TF_Graph*, const char*);
TF_Operation* TF_GraphNextOperation(TF_Graph*, size_t*);
int TF_GraphGetTensorNumDims(TF_Graph*, TF_Output, TF_Status*);
void TF_GraphGetTensorShape(TF_Graph*, TF_Output, int64_t*, int, TF_Status*);
void TF_GraphSetTensorShape(TF_Graph*, TF_Output, const int64_t*, int,
  TF_Status*);
const char* TF_OperationName(TF_Operation*);
const char* TF_OperationOpType(TF_Operation*);
const char* TF_OperationDevice(TF_Operation*);
int TF_OperationNumInputs(TF_Operation*);
void TF_OperationAllInputs(TF_Operation*, TF_Output*, int);
int TF_OperationInputListLength(TF_Operation*, const char*, TF_Status*);
int TF_OperationOutputListLength(TF_Operation*, const char*, TF_Status*);
TF_DataType TF_OperationInputType(TF_Input);
TF_Output TF_OperationInput(TF_Input);
int TF_OperationNumOutputs(TF_Operation*);
TF_DataType TF_OperationOutputType(TF_Output);
int TF_OperationOutputNumConsumers(TF_Output);
int TF_OperationOutputConsumers(TF_Output, TF_Input*, int);
int TF_OperationNumControlInputs(TF_Operation*);
int TF_OperationGetControlInputs(TF_Operation*, TF_Operation**, int);
int TF_OperationNumControlOutputs(TF_Operation*);
int TF_OperationGetControlOutputs(TF_Operation*, TF_Operation**, int);
TF_AttrMetadata TF_OperationGetAttrMetadata(TF_Operation*, const char*,
  TF_Status*);
int TF_OperationGetNumAttrs(TF_Operation*);
int TF_OperationGetAttrNameLength(TF_Operation*, int);
void TF_OperationGetAttrName(TF_Operation*, int, char*, TF_Status*);
void TF_OperationGetAttrString(TF_Operation*, const char*, void*, size_t,
  TF_Status*);
void TF_OperationGetAttrStringList(TF_Operation*, const char*, void**,
  size_t*, int, void*, size_t, TF_Status*);
void TF_OperationGetAttrInt(TF_Operation*, const char*, int64_t*, TF_Status*);
void TF_OperationGetAttrIntList(TF_Operation*, const char*, int64_t*, int,
  TF_Status*);
void TF_OperationGetAttrFloat(TF_Operation*, const char*, float*, TF_Status*);
void TF_OperationGetAttrFloatList(TF_Operation*, const char*, float*, int,
  TF_Status*);
void TF_OperationGetAttrBool(TF_Operation*, const char*, unsigned char*,
  TF_Status*);
void TF_OperationGetAttrBoolList(TF_Operation*, const char*, unsigned char*,
  int, TF_Status*);
void TF_OperationGetAttrType(TF_Operation*, const char*, TF_DataType*,
  TF_Status*);
void TF_OperationGetAttrTypeList(TF_Operation*, const char*, TF_DataType*, int,
  TF_Status*);
void TF_OperationGetAttrShape(TF_Operation*, const char*, int64_t*, int,
  TF_Status*);
void TF_OperationGetAttrShapeList(TF_Operation*, const char*, int64_t**, int*,
  int, int64_t*, int, TF_Status*);
void TF_OperationGetAttrTensor(TF_Operation*, const char*, TF_Tensor**,
  TF_Status*);
void TF_OperationGetAttrValueProto(TF_Operation*, const char*, TF_Buffer*,
  TF_Status*);
TF_OperationDescription* TF_NewOperation(TF_Graph*, const char*, const char*);
void TF_AddInput(TF_OperationDescription*, TF_Output);
void TF_AddInputList(TF_OperationDescription*, const TF_Output*, int);
void TF_AddControlInput(TF_OperationDescription*, TF_Operation*);
void TF_SetDevice(TF_OperationDescription*, const char*);
void TF_SetAttrString(TF_OperationDescription*, const char*, const void*, size_t);
void TF_SetAttrInt(TF_OperationDescription*, const char*, int64_t);
void TF_SetAttrFloat(TF_OperationDescription*, const char*, float);
void TF_SetAttrBool(TF_OperationDescription*, const char*, unsigned char);
void TF_SetAttrType(TF_OperationDescription*, const char*, TF_DataType);
void TF_SetAttrTypeList(TF_OperationDescription*, const char*, const TF_DataType*, int);
void TF_SetAttrIntList(TF_OperationDescription*, const char*, const int64_t*, int);
void TF_SetAttrShape(TF_OperationDescription*, const char*, const int64_t*, int);
void TF_SetAttrTensor(TF_OperationDescription*, const char*, TF_Tensor*, TF_Status*);
TF_Operation* TF_FinishOperation(TF_OperationDescription*, TF_Status*);
void TF_AddGradients(TF_Graph*, TF_Output* y, int ny, TF_Output* x, int nx,
                     TF_Output* dx, TF_Status*, TF_Output* dy);
void TF_GraphImportGraphDef(TF_Graph*, const TF_Buffer*,
                            const TF_ImportGraphDefOptions*, TF_Status*);
void TF_GraphToGraphDef(TF_Graph*, TF_Buffer*, TF_Status*);
void TF_OperationToNodeDef(TF_Operation*, TF_Buffer*, TF_Status*);
void TF_GraphGetOpDef(TF_Graph*, const char*, TF_Buffer*, TF_Status*);
void TF_GraphVersions(TF_Graph*, TF_Buffer*, TF_Status*);
const char* TF_GraphDebugString(TF_Graph*, size_t*);
TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions(void);
void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions*);
void TF_ImportGraphDefOptionsSetPrefix(TF_ImportGraphDefOptions*,
  const char*);
void TF_ImportGraphDefOptionsSetDefaultDevice(TF_ImportGraphDefOptions*,
  const char*);
void TF_ImportGraphDefOptionsSetUniquifyNames(TF_ImportGraphDefOptions*,
  unsigned char);
void TF_ImportGraphDefOptionsSetUniquifyPrefix(TF_ImportGraphDefOptions*,
  unsigned char);
void TF_ImportGraphDefOptionsAddInputMapping(TF_ImportGraphDefOptions*,
  const char*, int, TF_Output);

TF_Buffer* TF_NewBuffer(void);
TF_Buffer* TF_NewBufferFromString(const void*, size_t);
void TF_DeleteBuffer(TF_Buffer*);
TF_Buffer TF_GetBuffer(TF_Buffer*);

TF_SessionOptions* TF_NewSessionOptions(void);
void TF_DeleteSessionOptions(TF_SessionOptions*);
void TF_SetTarget(TF_SessionOptions*, const char*);
TF_Session* TF_NewSession(TF_Graph*, const TF_SessionOptions*, TF_Status*);
TF_Session* TF_LoadSessionFromSavedModel(const TF_SessionOptions*,
  const TF_Buffer*, const char*, const char* const*, int, TF_Graph*,
  TF_Buffer*, TF_Status*);
void TF_CloseSession(TF_Session*, TF_Status*);
void TF_DeleteSession(TF_Session*, TF_Status*);
void TF_SessionRun(TF_Session*, const TF_Buffer*,
  const TF_Output*, TF_Tensor* const*, int,
  const TF_Output*, TF_Tensor**, int,
  const TF_Operation* const*, int,
  TF_Buffer*, TF_Status*);

TF_CheckpointReader* TF_NewCheckpointReader(const char*, TF_Status*);
void TF_DeleteCheckpointReader(TF_CheckpointReader*);
int TF_CheckpointReaderHasTensor(TF_CheckpointReader*, const char*);
const char* TF_CheckpointReaderGetVariable(TF_CheckpointReader*, int);
int TF_CheckpointReaderSize(TF_CheckpointReader*);
TF_DataType TF_CheckpointReaderGetVariableDataType(TF_CheckpointReader*,
  const char*);
int TF_CheckpointReaderGetVariableNumDims(TF_CheckpointReader*, const char*);
void TF_CheckpointReaderGetVariableShape(TF_CheckpointReader*, const char*,
  int64_t*, int, TF_Status*);
TF_Tensor* TF_CheckpointReaderGetTensor(TF_CheckpointReader*, const char*,
  TF_Status*);

void TF_StringInit(TF_TString*);
void TF_StringCopy(TF_TString*, const char*, size_t);
const char* TF_StringGetDataPointer(const TF_TString*);
size_t TF_StringGetSize(const TF_TString*);
void TF_StringDealloc(TF_TString*);

TFE_ContextOptions* TFE_NewContextOptions(void);
void TFE_DeleteContextOptions(TFE_ContextOptions*);
void TFE_ContextOptionsSetAsync(TFE_ContextOptions*, unsigned char);
TFE_Context* TFE_NewContext(const TFE_ContextOptions*, TF_Status*);
void TFE_DeleteContext(TFE_Context*);
TFE_TensorHandle* TFE_NewTensorHandle(const TF_Tensor*, TF_Status*);
void TFE_DeleteTensorHandle(TFE_TensorHandle*);
TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle*);
int TFE_TensorHandleNumDims(TFE_TensorHandle*, TF_Status*);
int64_t TFE_TensorHandleDim(TFE_TensorHandle*, int, TF_Status*);
TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle*, TF_Status*);
TFE_Op* TFE_NewOp(TFE_Context*, const char*, TF_Status*);
void TFE_DeleteOp(TFE_Op*);
void TFE_OpAddInput(TFE_Op*, TFE_TensorHandle*, TF_Status*);
void TFE_OpSetAttrString(TFE_Op*, const char*, const void*, size_t);
void TFE_OpSetAttrInt(TFE_Op*, const char*, int64_t);
void TFE_OpSetAttrFloat(TFE_Op*, const char*, float);
void TFE_OpSetAttrBool(TFE_Op*, const char*, unsigned char);
void TFE_OpSetAttrType(TFE_Op*, const char*, TF_DataType);
void TFE_OpSetAttrIntList(TFE_Op*, const char*, const int64_t*, int);
void TFE_OpSetAttrShape(TFE_Op*, const char*, const int64_t*, int, TF_Status*);
void TFE_Execute(TFE_Op*, TFE_TensorHandle**, int*, TF_Status*);

TF_Tensor* TF_AllocateTensor(TF_DataType, const int64_t*, int, size_t);
TF_Tensor* TF_NewTensor(TF_DataType, const int64_t*, int, void*, size_t,
  void (*)(void*, size_t, void*), void*);
void TF_DeleteTensor(TF_Tensor*);
TF_DataType TF_TensorType(const TF_Tensor*);
int TF_NumDims(const TF_Tensor*);
int64_t TF_Dim(const TF_Tensor*, int);
size_t TF_TensorByteSize(const TF_Tensor*);
void* TF_TensorData(const TF_Tensor*);
int64_t TF_TensorElementCount(const TF_Tensor*);
size_t TF_DataTypeSize(TF_DataType);
]]

local capi = { ffi = ffi,
  released = { tensors = 0, graphs = 0, sessions = 0, readers = 0,
    eager_tensors = 0, contexts = 0 } }

local function candidates()
  local result = {}
  local explicit = os.getenv("TENSORFLOW_LIB")
  if explicit and explicit ~= "" then
    if explicit:match("%.so") then result[#result + 1] = explicit
    else result[#result + 1] = explicit .. "/libtensorflow.so" end
  end
  local root = os.getenv("TENSORFLOW_ROOT")
  if root and root ~= "" then
    result[#result + 1] = root .. "/lib/libtensorflow.so"
    result[#result + 1] = root .. "/libtensorflow/lib/libtensorflow.so"
  end
  local info = debug.getinfo(1, "S").source:gsub("^@", "")
  local source = info:match("^(.*)/tensorflow/capi.lua$") or "."
  result[#result + 1] = source .. "/third_party/libtensorflow/lib/libtensorflow.so"
  result[#result + 1] = "third_party/libtensorflow/lib/libtensorflow.so"
  result[#result + 1] = "tensorflow"
  return result
end

local function load_one(path)
  local framework = path:gsub("libtensorflow%.so$", "libtensorflow_framework.so")
  if framework ~= path then pcall(ffi.load, framework, true) end
  local ok, lib = pcall(ffi.load, path, true)
  if ok then return lib end
end

function capi.load()
  if capi.lib then return capi.lib end
  local errors = {}
  for _, path in ipairs(candidates()) do
    local lib = load_one(path)
    if lib then
      local ok, version = pcall(function() return ffi.string(lib.TF_Version()) end)
      if ok then
        capi.lib = lib
        capi.version = version
        return lib
      end
    end
    errors[#errors + 1] = path
  end
  error("cannot load libtensorflow; tried:\n  " ..
    table.concat(errors, "\n  ") ..
    "\nSet TENSORFLOW_LIB to the shared library or TENSORFLOW_ROOT to its root.", 2)
end

function capi.release_tensor(pointer)
  capi.released.tensors = capi.released.tensors + 1
  capi.load().TF_DeleteTensor(pointer)
end

function capi.release_graph(pointer)
  capi.released.graphs = capi.released.graphs + 1
  capi.load().TF_DeleteGraph(pointer)
end

function capi.release_reader(pointer)
  capi.released.readers = capi.released.readers + 1
  capi.load().TF_DeleteCheckpointReader(pointer)
end

function capi.require(pointer, what)
  if pointer == nil then
    error("TensorFlow allocation failed: " .. tostring(what), 2)
  end
  return pointer
end

function capi.status()
  return capi.require(capi.load().TF_NewStatus(), "TF_NewStatus")
end

function capi.check(status, context)
  local lib = capi.load()
  local code = lib.TF_GetCode(status)
  if code ~= 0 then
    local message_pointer = lib.TF_Message(status)
    local message = message_pointer ~= nil and
      ffi.string(message_pointer) or "unknown TensorFlow error"
    lib.TF_DeleteStatus(status)
    error((context and (context .. ": ") or "") .. message, 3)
  end
  lib.TF_DeleteStatus(status)
end

return capi
