//
// request_handler.cpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include "request_handler.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "mime_types.hpp"
#include "reply.hpp"
#include "request.hpp"

namespace http {
namespace server {

std::vector<std::string> strip_delimiter(const std::string& input, const std::string& delimiter) {
  std::vector<std::string> results;
  size_t last = 0;
  size_t next = 0;
  while ((next = input.find(delimiter, last)) != std::string::npos) {
    results.push_back(input.substr(last, next-last));
    last = next + 1;
  }
  results.push_back(input.substr(last));
  return results;
}

request_handler::request_handler(const std::string& doc_root)
  : doc_root_(doc_root)
{
}

void request_handler::handle_request(const request& req, reply& rep, volatile int& flag)
{
  // Decode url to path.
  std::string request_path;
  std::string extension;  // extension of rep content
  if (!url_decode(req.uri, request_path))
  {
    rep = reply::stock_reply(reply::bad_request);
    return;
  }
  if (req.method == "POST" || req.method == "post") {
    for (auto hdr : req.headers) {
      if (hdr.name == "Content-Type" &&
          hdr.value == "application/x-www-form-urlencoded") {
        std::vector<std::string> kv_strs = strip_delimiter(req.body, "&");
        for (std::string kv_str : kv_strs) {
          std::vector<std::string> kv = strip_delimiter(kv_str, "=");
          if (kv.size() == 2 && kv[0] == "run") {
            flag = std::stoi(kv[1]);
          }
        }
      }
    }
    extension = "text";
    rep.content = "Flag updated.\n";
  } else if (req.method == "GET" || req.method == "get") {
    // Request path must be absolute and not contain "..".
    if (request_path.empty() || request_path[0] != '/'
        || request_path.find("..") != std::string::npos)
    {
      rep = reply::stock_reply(reply::bad_request);
      return;
    }

    // If path ends in slash (i.e. is a directory) then add "index.html".
    if (request_path[request_path.size() - 1] == '/')
    {
      request_path += "index.html";
    }

    // Determine the file extension.
    std::size_t last_slash_pos = request_path.find_last_of("/");
    std::size_t last_dot_pos = request_path.find_last_of(".");
    if (last_dot_pos != std::string::npos && last_dot_pos > last_slash_pos)
    {
      extension = request_path.substr(last_dot_pos + 1);
    }

    // Open the file to send back.
    std::string full_path = doc_root_ + request_path;
    std::ifstream is(full_path.c_str(), std::ios::in | std::ios::binary);
    if (!is)
    {
      rep = reply::stock_reply(reply::not_found);
      return;
    }

    char buf[512];
    while (is.read(buf, sizeof(buf)).gcount() > 0)
      rep.content.append(buf, is.gcount());
  }
  // Fill out the reply to be sent to the client.
  rep.status = reply::ok;
  rep.headers.resize(2);
  rep.headers[0].name = "Content-Length";
  rep.headers[0].value = std::to_string(rep.content.size());
  rep.headers[1].name = "Content-Type";
  rep.headers[1].value = mime_types::extension_to_type(extension);
}

bool request_handler::url_decode(const std::string& in, std::string& out)
{
  out.clear();
  out.reserve(in.size());
  for (std::size_t i = 0; i < in.size(); ++i)
  {
    if (in[i] == '%')
    {
      if (i + 3 <= in.size())
      {
        int value = 0;
        std::istringstream is(in.substr(i + 1, 2));
        if (is >> std::hex >> value)
        {
          out += static_cast<char>(value);
          i += 2;
        }
        else
        {
          return false;
        }
      }
      else
      {
        return false;
      }
    }
    else if (in[i] == '+')
    {
      out += ' ';
    }
    else
    {
      out += in[i];
    }
  }
  return true;
}
} // namespace server
} // namespace http
