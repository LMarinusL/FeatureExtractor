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
    [SerializeField] public TextMeshProUGUI _valueText;
    public int receivedTime = 0;

    bool running;

    private void Update()
    {
        _valueText.text = receivedTime.ToString();
    }

    private void Start()
    {
        ThreadStart ts = new ThreadStart(GetInfo);
        mThread = new Thread(ts);
        mThread.Start();
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

        if (dataReceived != null)
        {
            receivedTime = StringToInt(dataReceived); 
            print("received time data");

            byte[] myWriteBuffer = Encoding.ASCII.GetBytes("Unity connected"); 
            nwStream.Write(myWriteBuffer, 0, myWriteBuffer.Length); 
        }
    }

    public static int StringToInt(string stime)
    {
        return int.Parse(stime);
    }
 
}