using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using UnityEngine.UI;
using TMPro;

// https://github.com/CanYouCatchMe01/CSharp-and-Python-continuous-communication


public class PythonConnection : MonoBehaviour
{

    Thread mThread;
    public string connectionIP = "127.0.0.1";
    public int connectionPort = 25001;
    IPAddress localAdd;
    TcpListener listener;
    TcpClient client;
    public int receivedTime = 0;
    public TextAsset vertexFile1997;
    public byte[] myWriteBuffer;

    bool running;

    private void Update()
    {
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
        myWriteBuffer = vertexFile1997.bytes;

    }

    void GetInfo()
    {
        localAdd = IPAddress.Parse(connectionIP);
        listener = new TcpListener(IPAddress.Any, connectionPort);
        listener.Start();

        client = listener.AcceptTcpClient();

        running = true;
        while (running)
        {
            SendAndReceiveData();
        }
        listener.Stop();
    }

    void SendAndReceiveData()
    {
        NetworkStream nwStream = client.GetStream();
        byte[] buffer = new byte[client.ReceiveBufferSize];

        int bytesRead = nwStream.Read(buffer, 0, client.ReceiveBufferSize); 
        string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);

        Encoding u8 = Encoding.UTF8;

        nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); 
        
    }

    public static int StringToInt(string stime)
    {
        return int.Parse(stime);
    }
 
}