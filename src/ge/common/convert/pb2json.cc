/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// File:        pb2json.h
// Description: This imply file for protobuf message and json interconversion

#include "common/convert/pb2json.h"

#include <set>
#include <string>

#include "framework/common/fmk_types.h"

using std::set;
using std::string;

namespace ge {
// JSON parses non utf8 character throwing exceptions, so some fields need to be shielded through black fields
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY void Pb2Json::Message2Json(const ProtobufMsg &message,
                                                                            const set<string> &black_fields, Json &json,
                                                                            bool enum2str) {
  auto descriptor = message.GetDescriptor();
  auto reflection = message.GetReflection();
  if (descriptor == nullptr || reflection == nullptr) {
    return;
  }

  auto count = descriptor->field_count();

  for (auto i = 0; i < count; ++i) {
    const auto field = descriptor->field(i);
    if (field == nullptr) {
      return;
    }

    // Do not display weight data
    if (black_fields.find(field->name()) != black_fields.end()) {
      continue;
    }

    if (field->is_repeated()) {
      if (reflection->FieldSize(message, field) > 0) {
        RepeatedMessage2Json(message, field, reflection, black_fields, json[field->name()], enum2str);
      }
      continue;
    }

    if (!reflection->HasField(message, field)) {
      continue;
    }

    OneField2Json(message, field, reflection, black_fields, json, enum2str);
  }
}

void Pb2Json::OneField2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                            const ProtobufReflection *reflection, const set<string> &black_fields, Json &json,
                            bool enum2str) {
  if (field == nullptr || reflection == nullptr) {
    return;
  }
  switch (field->type()) {
    case ProtobufFieldDescriptor::TYPE_MESSAGE: {
      const ProtobufMsg &tmp_message = reflection->GetMessage(message, field);
      if (0 != tmp_message.ByteSize()) {
        Message2Json(tmp_message, black_fields, json[field->name()]);
      }
      break;
    }

    case ProtobufFieldDescriptor::TYPE_BOOL:
      json[field->name()] = reflection->GetBool(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_ENUM: {
      auto *enum_value_desc = reflection->GetEnum(message, field);
      Enum2Json(enum_value_desc, field, enum2str, json);
      break;
    }

    case ProtobufFieldDescriptor::TYPE_INT32:
    case ProtobufFieldDescriptor::TYPE_SINT32:
    case ProtobufFieldDescriptor::TYPE_SFIXED32:
      json[field->name()] = reflection->GetInt32(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_UINT32:
    case ProtobufFieldDescriptor::TYPE_FIXED32:
      json[field->name()] = reflection->GetUInt32(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_INT64:
    case ProtobufFieldDescriptor::TYPE_SINT64:
    case ProtobufFieldDescriptor::TYPE_SFIXED64:
      json[field->name()] = reflection->GetInt64(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_UINT64:
    case ProtobufFieldDescriptor::TYPE_FIXED64:
      json[field->name()] = reflection->GetUInt64(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_FLOAT:
      json[field->name()] = reflection->GetFloat(message, field);
      break;

    case ProtobufFieldDescriptor::TYPE_STRING:
    case ProtobufFieldDescriptor::TYPE_BYTES:
      json[field->name()] = reflection->GetString(message, field);
      break;

    default:
      break;
  }
}

void Pb2Json::RepeatedMessage2Json(const ProtobufMsg &message, const ProtobufFieldDescriptor *field,
                                   const ProtobufReflection *reflection, const set<string> &black_fields, Json &json,
                                   bool enum2str) {
  if (field == nullptr || reflection == nullptr) {
    Message2Json(message, black_fields, json);
    return;
  }

  for (auto i = 0; i < reflection->FieldSize(message, field); ++i) {
    Json tmp_json;
    switch (field->type()) {
      case ProtobufFieldDescriptor::TYPE_MESSAGE: {
        const ProtobufMsg &tmp_message = reflection->GetRepeatedMessage(message, field, i);
        if (0 != tmp_message.ByteSize()) {
          Message2Json(tmp_message, black_fields, tmp_json);
        }
      } break;

      case ProtobufFieldDescriptor::TYPE_BOOL:
        tmp_json = reflection->GetRepeatedBool(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_ENUM: {
        auto *enum_value_desc = reflection->GetRepeatedEnum(message, field, i);
        RepeatedEnum2Json(enum_value_desc, enum2str, tmp_json);
      } break;

      case ProtobufFieldDescriptor::TYPE_INT32:
      case ProtobufFieldDescriptor::TYPE_SINT32:
      case ProtobufFieldDescriptor::TYPE_SFIXED32:
        tmp_json = reflection->GetRepeatedInt32(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_UINT32:
      case ProtobufFieldDescriptor::TYPE_FIXED32:
        tmp_json = reflection->GetRepeatedUInt32(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_INT64:
      case ProtobufFieldDescriptor::TYPE_SINT64:
      case ProtobufFieldDescriptor::TYPE_SFIXED64:
        tmp_json = reflection->GetRepeatedInt64(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_UINT64:
      case ProtobufFieldDescriptor::TYPE_FIXED64:
        tmp_json = reflection->GetRepeatedUInt64(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_FLOAT:
        tmp_json = reflection->GetRepeatedFloat(message, field, i);
        break;

      case ProtobufFieldDescriptor::TYPE_STRING:
      case ProtobufFieldDescriptor::TYPE_BYTES:
        tmp_json = reflection->GetRepeatedString(message, field, i);
        break;

      default:
        break;
    }
    json += tmp_json;
  }
}

void Pb2Json::Enum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, const ProtobufFieldDescriptor *field,
                        bool enum2str, Json &json) {
  if (enum_value_desc != nullptr) {
    if (field == nullptr) {
      return;
    }
    if (enum2str) {
      json[field->name()] = enum_value_desc->name();
    } else {
      json[field->name()] = enum_value_desc->number();
    }
  }
}

void Pb2Json::RepeatedEnum2Json(const ProtobufEnumValueDescriptor *enum_value_desc, bool enum2str, Json &json) {
  if (enum_value_desc != nullptr) {
    if (enum2str) {
      json = enum_value_desc->name();
    } else {
      json = enum_value_desc->number();
    }
  }
}
}  //  namespace ge
