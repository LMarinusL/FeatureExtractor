using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System.Linq;
using System;
using Unity.Mathematics;
public class ContourGenerator : MonoBehaviour
{

    public Grid grid;
    public int xSize;
    public int zSize;
    public GameObject dotgreen;
    public Vector3[] vectorList;


    void Update()
    {
        if (Input.GetKeyDown(KeyCode.H))
        {
            getData();
            getContourVertices(100f);
        }
        
    }

    void getData()
    {
        GameObject gridcreator = GameObject.Find("GridCreator");
        CreateGrid gridImporter = gridcreator.GetComponent<CreateGrid>();

        grid = gridImporter.grid;
        xSize = gridImporter.xSize;
        zSize = gridImporter.zSize;

        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
    }

    void getContourVertices(float height)
    {
        List<float3> contourVertices = new List<float3>();
        Cell currentCell = grid.cells[0];
        Cell previousCell;
        float xStep = grid.cells[4 + xSize + 1].x - grid.cells[4].x;
        float zStep = grid.cells[4 + xSize + 1].z - grid.cells[4].z;
        foreach (Cell cell in grid.cells)
        {
            previousCell = currentCell;
            currentCell = cell;
            if (currentCell.y == 0 || previousCell.y == 0 ||
                Mathf.Pow(Mathf.Pow(currentCell.x - previousCell.x, 2) + Mathf.Pow(currentCell.z - previousCell.z,2), 0.5f) > 2*xStep) { continue; }

            
            if((currentCell.y <= height && previousCell.y > height) ||
                (currentCell.y >= height && previousCell.y < height))
            {
                float ratio = ((currentCell.y - height) / (previousCell.y - height));
                contourVertices.Add(new float3(currentCell.x + (xStep * ratio), height,
                    currentCell.z));
            }
        }
        Debug.Log("contour: "+ contourVertices.ToArray().Length);
        GameObject terrain = GameObject.Find("TerrainLoader");
        MeshGenerator meshGenerator = terrain.GetComponent<MeshGenerator>();
        vectorList = meshGenerator.float3ToVector3Array(contourVertices.ToArray());
        foreach (Vector3 point in vectorList)
        {
            Instantiate(dotgreen, point, transform.rotation);
        }

    }
}
