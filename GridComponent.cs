using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEditor;
using System.Linq;
using System;
using Unity.Mathematics;

public class Grid : Component
{ // list of grid cells in same grid order as input cells
    public float3[] OriginalPC;
    public Cell[] cells;
    public Triangle[] triangles;

    public Grid(float3[] originalPC, float3[] originalNormals, int[] triangleMesh)
    {
        OriginalPC = originalPC;
        cells = new Cell[originalPC.Length];
        triangles = new Triangle[triangleMesh.Length];
        for (int i = 0; i < originalPC.Length; i++)
        {
            cells[i] = new Cell(i, originalPC[i], originalNormals[i]);
        }
        setTriangles(triangleMesh);
        setTwins();
    }

    public void setTriangles(int[] trianglesInput)
    {
        int j = 0;
        for (int i = 0; i < trianglesInput.Length; i += 3)
        {
            int[] tri = new int[3] { trianglesInput[i], trianglesInput[i + 1], trianglesInput[i + 2] };
            triangles[j] = new Triangle(j, tri, cells, this);
            j++;
        }
    }
    public void setTwins()
    {
        for (int i = 0; i < cells.Length; i++)
        {
            foreach (Face face in cells[i].attachedFaces)
            {
                int start1 = face.startVertex.index;
                int end1 = face.endVertex.index;

                if (face.faceTwin == null)
                {
                    foreach (Face face2 in cells[i].attachedFaces)
                    {
                        int end2 = face2.endVertex.index;
                        int start2 = face2.startVertex.index;
                        if (face2.faceTwin == null && start1 == end2 && start2 == end1)
                        {
                            face.faceTwin = face2;
                            face2.faceTwin = face;
                        }
                    }
                }
            }
        }
    }

}

public class Triangle : Component
{
    public int index;
    public Cell[] vertices;
    public Face[] faces;

    public Triangle(int i, int[] vertindex, Cell[] cells, Grid grid)
    {
        index = i;
        vertices = new Cell[vertindex.Length];
        for (int p = 0; p < vertindex.Length; p++)
        {
            vertices[p] = grid.cells[vertindex[p]];
        }
        faces = new Face[3];
        faces[0] = new Face((i * 10) + 1, vertices[0], vertices[1], this, grid);
        faces[1] = new Face((i * 10) + 2, vertices[1], vertices[2], this, grid);
        faces[2] = new Face((i * 10) + 3, vertices[2], vertices[0], this, grid);


        for (int k = 0; k < vertices.Length; k++)
        {
            vertices[k].attachedFaces.Add(faces[k]);
            vertices[k].attachedTriangles.Add(this);
        }

        vertices[0].attachedFaces.Add(faces[2]);
        vertices[1].attachedFaces.Add(faces[0]);
        vertices[2].attachedFaces.Add(faces[1]);



    }

}

public class Face : Component
{
    public int index;
    public Cell startVertex;
    public Cell endVertex;
    public Triangle ownTriangle;
    public Face faceTwin;
    public int onContourLine;




    public Face(int i, Cell start, Cell end, Triangle own, Grid grid)
    {
        index = i;
        startVertex = start;
        endVertex = end;
        ownTriangle = own;
        onContourLine = 0;
    }

    public Face next()
    {
        Face nextFace = null;
        Face[] faces = this.ownTriangle.faces;
        foreach (Face face in faces)
        {
            if (face.startVertex.index == this.endVertex.index)
            {
                nextFace = face;
            }
        }
        return nextFace;
    }

    public Face previous()
    {
        Face previousFace = null;
        Face[] faces = this.ownTriangle.faces;
        foreach (Face face in faces)
        {
            if (face.endVertex.index == this.startVertex.index)
            {
                previousFace = face;
            }
        }
        return previousFace;
    }
}

public class Cell : Component
{
    public int index;
    public Vector3 position;
    public float x;
    public float y;
    public float z;
    public float slope;
    public float aspect;
    public float relativeHeight1;
    public float relativeHeight2;
    public float relativeHeight3;
    public float relativeSlope;
    public float relativeAspect;
    public float dRM1;
    public float dLN1;
    public float curvatureX;
    public float curvatureY;
    public float curvatureZ;
    public float curvature;
    public int runoffScore;
    public float averageRunoff1;
    public float averageRunoff2;
    public float averageRunoff3;
    public float distToSkeleton;
    public List<Triangle> attachedTriangles;
    public List<Face> attachedFaces;
    public ContourCell contourCell;
    public float skeletonAspect;
    public float distToRiverMouth;
    public float riverDischarge;



    public Cell(int i, float3 loc, float3 normal)
    {
        index = i;
        x = loc.x;
        y = loc.y;
        z = loc.z;
        position = new Vector3(loc.x, loc.y, loc.z);
        slope = computeSlope(normal);
        aspect = computeAspect(normal);
        runoffScore = 0;
        attachedTriangles = new List<Triangle>();
        attachedFaces = new List<Face>();
        averageRunoff1 = 0;
        averageRunoff2 = 0;
        averageRunoff3 = 0;

    }

    float computeSlope(float3 normal)
    {
        float slope = Mathf.Atan(Mathf.Pow((Mathf.Pow(normal.x, 2f) + Mathf.Pow(normal.z, 2f)), 0.5f) / normal.y);
        return slope;
    }

    float computeAspect(float3 normal)
    {
        float aspect;
        if (normal.x > 0)
        {
            aspect = 90f - 57.3f * (Mathf.Atan(normal.z / normal.x));
        }
        else
        {
            aspect = 270f - 57.3f * (Mathf.Atan(normal.z / normal.x));
        }
        return aspect;
    }
}

