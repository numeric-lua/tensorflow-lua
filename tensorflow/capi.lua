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

const char* TF_Version(void);
TF_Status* TF_NewStatus(void);
void TF_DeleteStatus(TF_Status*);
TF_Code TF_GetCode(const TF_Status*);
const char* TF_Message(const TF_Status*);

TF_Graph* TF_NewGraph(void);
void TF_DeleteGraph(TF_Graph*);
TF_Operation* TF_GraphOperationByName(TF_Graph*, const char*);
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
void TF_SetAttrShape(TF_OperationDescription*, const char*, const int64_t*, int);
void TF_SetAttrTensor(TF_OperationDescription*, const char*, TF_Tensor*, TF_Status*);
TF_Operation* TF_FinishOperation(TF_OperationDescription*, TF_Status*);
void TF_AddGradients(TF_Graph*, TF_Output* y, int ny, TF_Output* x, int nx,
                     TF_Output* dx, TF_Status*, TF_Output* dy);
void TF_GraphImportGraphDef(TF_Graph*, const TF_Buffer*,
                            const TF_ImportGraphDefOptions*, TF_Status*);
void TF_GraphToGraphDef(TF_Graph*, TF_Buffer*, TF_Status*);
TF_ImportGraphDefOptions* TF_NewImportGraphDefOptions(void);
void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions*);

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
  released = { tensors = 0, graphs = 0, sessions = 0, readers = 0 } }

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
