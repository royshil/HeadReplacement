#include "stdafx.h"
/*****************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * $Id: ftpget.c,v 1.8 2008-05-22 21:20:09 danf Exp $
 */

#include <stdio.h>

#include <curl/curl.h>
#include <curl/types.h>
#include <curl/easy.h>

/*
 * This is an example showing how to get a single file from an FTP server.
 * It delays the actual destination file creation until the first write
 * callback so that it won't create an empty file in case the remote file
 * doesn't exist or something else fails.
 */

struct FtpFile {
  const char *filename;
  FILE *stream;
};

static std::string content_type = string();

static size_t header_handle( void *ptr, size_t size, size_t nmemb, void *stream) {
	string header_s((char*)ptr,size*nmemb);
	if(header_s.find("Content-Type") == 0) {
		int spaceidx = header_s.find_first_of(" ");
		int stopidx = min(header_s.find_first_of(";",spaceidx),header_s.length());
		content_type = header_s.substr(spaceidx+1,stopidx-spaceidx-1);
	}
	return size * nmemb;
}

static size_t my_fwrite(void *buffer, size_t size, size_t nmemb, void *stream)
{
  struct FtpFile *out=(struct FtpFile *)stream;
  if(out && !out->stream) {
    /* open file for writing */
	//errno_t err;
	//if((err = fopen_s(&(out->stream),out->filename, "wb")) != 0) {
	//	fprintf(stderr,strerror(err));
	out->stream = fopen(out->filename,"wb");
	if(!out->stream) {
      return -1; /* failure, can't open file to write */
	}
  }
  return fwrite(buffer, size, nmemb, out->stream);
}

/**
This function will go to the URL in url arg and save the file locally to the disk (at working directory)
It will put the saved filename in filename arg, unless filename arg already contains a target filename.
**/
int curl_get(std::string& url, std::string filename = "")
{
  CURL *curl;
  CURLcode res;

  if(filename.size() == 0) { //derive output from url..
	  int p = url.find_last_of("/");
	  filename = url.substr(p+1);
  }

  struct FtpFile ftpfile={
	  filename.c_str(), /* name to store the file as if succesful */
    NULL
  };

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if(curl) {
    /*
     * Get curl 7.9.2 from sunet.se's FTP site. curl 7.9.2 is most likely not
     * present there by the time you read this, so you'd better replace the
     * URL with one that works!
     */
    curl_easy_setopt(curl, CURLOPT_URL,	url.c_str());
    /* Define our callback to get called when there's data to be written */
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, my_fwrite);
    /* Set a pointer to our struct to pass to the callback */
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ftpfile);
   
	
	struct curl_slist *chunk = NULL;
    chunk = curl_slist_append(chunk, "User-Agent: Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/525.13 (KHTML, like Gecko) Chrome/0.A.B.C Safari/525.13");
    res = curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);

	/* Switch on full protocol/debug output */
    //curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

	//handle headers...
	curl_easy_setopt(curl,CURLOPT_HEADERFUNCTION,&header_handle);

    res = curl_easy_perform(curl);

    /* always cleanup */
    curl_easy_cleanup(curl);

    if(CURLE_OK != res) {
      /* we failed */
      fprintf(stderr, "curl told us %d\n", res);
    }
  }

	//TODO: if we don't have an extension for the file name, use the content-type header

  if(ftpfile.stream)
    fclose(ftpfile.stream); /* close the local file */

  curl_global_cleanup();

  return (CURLE_OK == res);
}
