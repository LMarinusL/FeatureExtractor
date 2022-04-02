using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Skeleton : Component
{
    public List<SkeletonJoint> skeletonC;
    public List<SkeletonJoint> skeletonP;

    public List<SkeletonJoint> VectorToJoints(List<Vector3> input, float discharge)
    {
        float distance = 0f;
        Vector3 previous = input[0];
        List<SkeletonJoint> newList = new List<SkeletonJoint>();
        foreach(Vector3 vertex in input)
        {
            distance += Vector3.Distance(previous, vertex);
            newList.Add(new SkeletonJoint(vertex, distance, discharge));
            previous = vertex;
        }
        return newList;
    }

    public Skeleton()
    {
        skeletonC = new List<SkeletonJoint>();
        skeletonP = new List<SkeletonJoint>();

    }

    public void addListC(List<Vector3> inputList, float discharge)
    {
        skeletonC = VectorToJoints(inputList, discharge);
    }
    public void addListP(List<Vector3> inputList, float discharge)
    {
        skeletonP = VectorToJoints(inputList, discharge);
    }

}


public class SkeletonJoint : Component
{
    public Vector3 position;
    public float distance;
    public float discharge;

    public SkeletonJoint(Vector3 pos, float dis, float disch)
    {
        position = pos;
        distance = dis;
        discharge = disch;
    }

}