using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class ShrinkingBallSeg : MonoBehaviour
{
    public List<Vector3> vertices;
    public List<Vector3> normals;
    public float initialRadius = 200.0f;
    public List<Vector3> verticesThin;
    public List<Vector3> MedialBallCenters;
    public List<float> MedialBallRadii;
    public GameObject dotred;
    public GameObject dotblue;
    MeshComponent meshComp;

    void Start()
    {

    }

    void Update()
    {
        if (Input.GetKey(KeyCode.I))
        {
            getMesh();
            meshComp = new MeshComponent(vertices);
            iterateVertices();
            InstantiatePoints();
        }
    }

    void getMesh()
    {
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        Mesh mesh = meshGenerator.mesh;
        vertices = mesh.vertices.ToList();
        normals = mesh.normals.ToList();
    }

    bool checkRadius(int vertexIndex, float radius)
    {
        Vector3 medialBallCenter = vertices[vertexIndex] + normals[vertexIndex] * radius;
        List<Vector3> list = meshComp.checkSegment(medialBallCenter, radius);
        foreach (Vector3 vertex in list)
        {
            if (Vector3.Distance(vertex, medialBallCenter) < radius)
            {
                return false;
            }
        }
        return true;
    }

    void getMedialBallCenter(int vertexIndex)
    {
        bool empty = false;
        float radius = initialRadius;
        while (empty == false)
        {
            radius -= 3.0f;
            empty = checkRadius(vertexIndex, radius);
            if (radius < 10f)
            {
                return;
            }
        }
        MedialBallCenters.Add((vertices[vertexIndex] + normals[vertexIndex] * radius));
        MedialBallRadii.Add(radius);
    }

    void iterateVertices()
    {
        for (int i = 0; i < vertices.ToArray().Length; i++)
        {
            if (vertices[i].y != 0f && normals[i].y>0.4f)
            {
                getMedialBallCenter(i);
            }
        }
    }
    void filterOnRadius()
    {

    }
    void InstantiatePoints()
    {
        MATList list = new MATList(MedialBallCenters);
        for (int vertId = 0; vertId < MedialBallCenters.ToArray().Length; vertId++)
        {
            Instantiate(dotblue, list.getLoc3D(vertId), transform.rotation);
            Instantiate(dotred, list.getLoc2D(vertId), transform.rotation);
        }
    }
}

public class MATList : Component
{
    public List<Vector3> OriginalMATList;
    public List<MATBall> NewMATList = new List<MATBall>();


    public MATList(List<Vector3> originalMATList) // constructor
    {
        OriginalMATList = originalMATList;
        foreach (Vector3 ball in OriginalMATList)
        {
            NewMATList.Add(new MATBall(ball));
        }
    }

    public Vector3 getLoc3D(int index)
    {
        return NewMATList[index].Loc;
    }
    public Vector3 getLoc2D(int index)
    {
        return new Vector3(NewMATList[index].Loc.x, 300, NewMATList[index].Loc.z);
    }
}

public class MATBall : Component
{
    public Vector3 Loc;

    public MATBall(Vector3 ballLoc) // constructor
    {
        Loc = ballLoc;
    }
}

