using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class ShrinkingBallSeg : MonoBehaviour
{
    public List<Vector3> vertices;
    public List<Vector3> normals;
    public float initialRadius = 250.0f;
    public List<Vector3> verticesThin;
    public List<Vector3> MedialBallCenters;
    public List<float> MedialBallRadii;
    public GameObject dotred;
    public GameObject dotblue;
    MeshComponent meshComp;

    /* to do: 
     - Create component class for medial balls to save properties
     - Filter the 2D set
     - Create component for medial branches
     - Compute medial branches 
    */

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
            radius -= 5.0f;
            empty = checkRadius(vertexIndex, radius);
            if (radius < 50f)
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
            getMedialBallCenter(i);
        }
    }
    void filterOnRadius()
    {

    }
    void InstantiatePoints()
    {
        for (int vertId = 0; vertId < MedialBallCenters.ToArray().Length; vertId++)
        {
            Instantiate(dotblue, MedialBallCenters[vertId], transform.rotation);
            Instantiate(dotred, new Vector3(MedialBallCenters[vertId].x, 300, MedialBallCenters[vertId].z), transform.rotation);
        }
    }
}
