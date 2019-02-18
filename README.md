This code works in conjunction with ffserver configured to stream mjpeg:

        HTTPPort            8090
        HTTPBindAddress     0.0.0.0
        MaxHTTPConnections 200
        MaxClients      100
        MaxBandWidth    500000
        CustomLog       -

        <Feed camera.ffm>
        File            /tmp/camera.ffm
        FileMaxSize     200M
        </Feed>

        <Stream camera.mjpeg>
        Feed camera.ffm
        Format mpjpeg
        VideoFrameRate 15
        VideoIntraOnly
        VideoBitRate 4096
        VideoBufferSize 4096
        VideoSize 640x480
        VideoQMin 5
        VideoQMax 51
        NoAudio
        Strict -1
        </Stream>

This code needs to be proxied by any web server with the following rules:

Example with nginx

        location /lv {
                proxy_pass  http://127.0.0.1:8090/camera.mjpeg;   #THE URL OF FFSERVER STREAM
        }
        location /chnl {
                proxy_pass  http://127.0.0.1:3000/;               #THE URL OF NODE PROCESS
        }
