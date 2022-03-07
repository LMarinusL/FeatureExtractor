using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Skeleton : Component
{
    public List<List<SkeletonJoint>> skeleton1997A;
    public List<List<SkeletonJoint>> skeleton2008A;
    public List<List<SkeletonJoint>> skeleton2012A;
    public List<List<SkeletonJoint>> skeleton2018A;
    public List<List<SkeletonJoint>> skeleton1997B;
    public List<List<SkeletonJoint>> skeleton2008B;
    public List<List<SkeletonJoint>> skeleton2012B;
    public List<List<SkeletonJoint>> skeleton2018B;

    // 1997
    List<Vector3> chagres1997 = new List<Vector3>
        {
           new Vector3(696, 0, 961),
           new Vector3(716, 0, 920),
           new Vector3(726, 0, 891),
           new Vector3(736, 0, 859),
           new Vector3(747, 0, 827),
           new Vector3(750, 0, 810),
           new Vector3(737, 0, 794),
           new Vector3(739, 0, 780),
           new Vector3(701, 0, 740),
           new Vector3(696, 0, 717),
           new Vector3(704, 0, 697),
           new Vector3(726, 0, 679),
           new Vector3(731, 0, 646),
           new Vector3(751, 0, 624),
           new Vector3(756, 0, 610),
           new Vector3(764, 0, 600),
           new Vector3(742, 0, 656),
        };

    List<Vector3> pequeni1997 = new List<Vector3>
        {
           new Vector3(1837, 0, 798),
           new Vector3(1811, 0, 787),
           new Vector3(1782, 0, 764),
           new Vector3(1757, 0, 752),
           new Vector3(1708, 0, 745),
           new Vector3(1689, 0, 734),
           new Vector3(1651, 0, 700),
           new Vector3(1605, 0, 678),
           new Vector3(1569, 0, 648),
           new Vector3(1559, 0, 618),
           new Vector3(1535, 0, 612),
           new Vector3(1513, 0, 637),
           new Vector3(1494, 0, 632),
           new Vector3(1485, 0, 584),
           new Vector3(1471, 0, 584),
           new Vector3(1448, 0, 618),
           new Vector3(1430, 0, 648),
           new Vector3(1421, 0, 646),
           new Vector3(1409, 0, 610),
           new Vector3(1404, 0, 585),
           new Vector3(1404, 0, 564),
           new Vector3(1350, 0, 557),
           new Vector3(1316, 0, 576),
           new Vector3(1272, 0, 620),
           new Vector3(1260, 0, 612),
           new Vector3(1250, 0, 598),
           new Vector3(1206, 0, 557),
           new Vector3(1183, 0, 557),
           new Vector3(1157, 0, 596),
           new Vector3(1105, 0, 637),
           new Vector3(1048, 0, 590),
           new Vector3(1013, 0, 558),
           new Vector3(994, 0, 568),
           new Vector3(945, 0, 615),
           new Vector3(938, 0, 646),
           new Vector3(901, 0, 643),
           new Vector3(864, 0, 633),
           new Vector3(852, 0, 611),
           new Vector3(844, 0, 593),
           new Vector3(825, 0, 592),
           new Vector3(799, 0, 596),
           new Vector3(777, 0, 575),
           new Vector3(765, 0, 568),
           new Vector3(736, 0, 568),
           new Vector3(706, 0, 588),
           new Vector3(658, 0, 598),
           new Vector3(622, 0, 584),
           new Vector3(561, 0, 587),
           new Vector3(501, 0, 587),
           new Vector3(447, 0, 601),
           new Vector3(433, 0, 555),
           new Vector3(445, 0, 530),
           new Vector3(481, 0, 525),
           new Vector3(491, 0, 506),
           new Vector3(535, 0, 485),
           new Vector3(550, 0, 442),
           new Vector3(550, 0, 388),
           new Vector3(575, 0, 388),
           new Vector3(613, 0, 418),
           new Vector3(624, 0, 403),
           new Vector3(627, 0, 362),
           new Vector3(637, 0, 343),
           new Vector3(627, 0, 324),
           new Vector3(598, 0, 342),
           new Vector3(569, 0, 342),
           new Vector3(579, 0, 308),
           new Vector3(617, 0, 293),
           new Vector3(633, 0, 268),
           new Vector3(629, 0, 257),
           new Vector3(586, 0, 252),
           new Vector3(535, 0, 227),
           new Vector3(535, 0, 248),
           new Vector3(551, 0, 266),
           new Vector3(551, 0, 288),
           new Vector3(529, 0, 295),
           new Vector3(418, 0, 277),
        };
    // 2008
    List<Vector3> chagres2008 = new List<Vector3>
        {
           new Vector3(696, 0, 961),
           new Vector3(717, 0, 915),
           new Vector3(747, 0, 849),
           new Vector3(769, 0, 829),
           new Vector3(761, 0, 803),
           new Vector3(765, 0, 795),
           new Vector3(743, 0, 762),
           new Vector3(747, 0, 743),
           new Vector3(722, 0, 727),
           new Vector3(714, 0, 692),
           new Vector3(731, 0, 669),
           new Vector3(731, 0, 649),
           new Vector3(765, 0, 609),
           new Vector3(758, 0, 572),
           new Vector3(734, 0, 572),
           new Vector3(696, 0, 591),
        };

    List<Vector3> pequeni2008 = new List<Vector3>
        {
           new Vector3(1837, 0, 798),
           new Vector3(1807, 0, 785),
           new Vector3(1733, 0, 730),
           new Vector3(1733, 0, 701),
           new Vector3(1696, 0, 653),
           new Vector3(1677, 0, 657),
           new Vector3(1644, 0, 656),
           new Vector3(1610, 0, 650),
           new Vector3(1610, 0, 630),
           new Vector3(1535, 0, 612),
           new Vector3(1513, 0, 637),
           new Vector3(1494, 0, 632),
           new Vector3(1485, 0, 584),
           new Vector3(1471, 0, 584),
           new Vector3(1448, 0, 618),
           new Vector3(1430, 0, 648),
           new Vector3(1421, 0, 646),
           new Vector3(1409, 0, 610),
           new Vector3(1404, 0, 585),
           new Vector3(1404, 0, 564),
           new Vector3(1350, 0, 557),
           new Vector3(1316, 0, 576),
           new Vector3(1272, 0, 620),
           new Vector3(1260, 0, 612),
           new Vector3(1250, 0, 598),
           new Vector3(1206, 0, 557),
           new Vector3(1183, 0, 557),
           new Vector3(1157, 0, 596),
           new Vector3(1105, 0, 637),
           new Vector3(1048, 0, 590),
           new Vector3(1013, 0, 558),
           new Vector3(994, 0, 568),
           new Vector3(945, 0, 615),
           new Vector3(938, 0, 646),
           new Vector3(901, 0, 643),
           new Vector3(864, 0, 633),
           new Vector3(852, 0, 611),
           new Vector3(844, 0, 593),
           new Vector3(809, 0, 602),
           new Vector3(762, 0, 595),
           new Vector3(762, 0, 571),
           new Vector3(736, 0, 568),
           new Vector3(706, 0, 588),
           new Vector3(658, 0, 598),
           new Vector3(622, 0, 584),
           new Vector3(561, 0, 587),
           new Vector3(501, 0, 587),
           new Vector3(447, 0, 601),
           new Vector3(433, 0, 555),
           new Vector3(445, 0, 530),
           new Vector3(481, 0, 525),
           new Vector3(491, 0, 506),
           new Vector3(535, 0, 485),
           new Vector3(550, 0, 442),
           new Vector3(550, 0, 388),
           new Vector3(575, 0, 388),
           new Vector3(613, 0, 418),
           new Vector3(624, 0, 403),
           new Vector3(627, 0, 362),
           new Vector3(637, 0, 343),
           new Vector3(627, 0, 324),
           new Vector3(598, 0, 342),
           new Vector3(569, 0, 342),
           new Vector3(579, 0, 308),
           new Vector3(617, 0, 293),
           new Vector3(633, 0, 268),
           new Vector3(629, 0, 257),
           new Vector3(586, 0, 252),
           new Vector3(535, 0, 227),
           new Vector3(535, 0, 248),
           new Vector3(551, 0, 266),
           new Vector3(551, 0, 288),
           new Vector3(529, 0, 295),
           new Vector3(418, 0, 277),
        };

    // 2012
    List<Vector3> chagres2012 = new List<Vector3>
        {
           new Vector3(696, 0, 961),
           new Vector3(724, 0, 900),
           new Vector3(661, 0, 844),
           new Vector3(671, 0, 827),
           new Vector3(664, 0, 814),
           new Vector3(672, 0, 746),
           new Vector3(730, 0, 672),
           new Vector3(730, 0, 655),
           new Vector3(763, 0, 608),
           new Vector3(756, 0, 574),
           new Vector3(734, 0, 571),
           new Vector3(707, 0, 585),
        };

    List<Vector3> pequeni2012 = new List<Vector3>
        {
           new Vector3(1834, 0, 813),
           new Vector3(1815, 0, 796),
           new Vector3(1780, 0, 760),
           new Vector3(1724, 0, 729),
           new Vector3(1664, 0, 662),
           new Vector3(1615, 0, 648),
           new Vector3(1608, 0, 628),
           new Vector3(1537, 0, 617),
           new Vector3(1513, 0, 637),
           new Vector3(1494, 0, 632),
           new Vector3(1485, 0, 584),
           new Vector3(1471, 0, 584),
           new Vector3(1448, 0, 618),
           new Vector3(1430, 0, 648),
           new Vector3(1421, 0, 646),
           new Vector3(1409, 0, 610),
           new Vector3(1404, 0, 585),
           new Vector3(1404, 0, 564),
           new Vector3(1350, 0, 557),
           new Vector3(1316, 0, 576),
           new Vector3(1272, 0, 620),
           new Vector3(1260, 0, 612),
           new Vector3(1250, 0, 598),
           new Vector3(1206, 0, 557),
           new Vector3(1183, 0, 557),
           new Vector3(1157, 0, 596),
           new Vector3(1105, 0, 637),
           new Vector3(1048, 0, 590),
           new Vector3(1013, 0, 558),
           new Vector3(994, 0, 568),
           new Vector3(945, 0, 615),
           new Vector3(938, 0, 646),
           new Vector3(901, 0, 643),
           new Vector3(864, 0, 647),
           new Vector3(833, 0, 643),
           new Vector3(833, 0, 615),
           new Vector3(820, 0, 600),
           new Vector3(799, 0, 596),
           new Vector3(777, 0, 575),
           new Vector3(765, 0, 568),
           new Vector3(736, 0, 568),
           new Vector3(706, 0, 588),
           new Vector3(658, 0, 598),
           new Vector3(622, 0, 584),
           new Vector3(561, 0, 587),
           new Vector3(501, 0, 587),
           new Vector3(447, 0, 601),
           new Vector3(433, 0, 555),
           new Vector3(445, 0, 530),
           new Vector3(481, 0, 525),
           new Vector3(491, 0, 506),
           new Vector3(535, 0, 485),
           new Vector3(550, 0, 442),
           new Vector3(550, 0, 388),
           new Vector3(575, 0, 388),
           new Vector3(613, 0, 418),
           new Vector3(624, 0, 403),
           new Vector3(627, 0, 362),
           new Vector3(637, 0, 343),
           new Vector3(627, 0, 324),
           new Vector3(598, 0, 342),
           new Vector3(569, 0, 342),
           new Vector3(579, 0, 308),
           new Vector3(617, 0, 293),
           new Vector3(633, 0, 268),
           new Vector3(629, 0, 257),
           new Vector3(586, 0, 252),
           new Vector3(535, 0, 227),
           new Vector3(535, 0, 248),
           new Vector3(551, 0, 266),
           new Vector3(551, 0, 288),
           new Vector3(529, 0, 295),
           new Vector3(418, 0, 277),
        };

        // 2018
    List<Vector3> chagres2018 = new List<Vector3>
        {
           new Vector3(709, 0, 935),
           new Vector3(721, 0, 889),
           new Vector3(685, 0, 837),
           new Vector3(673, 0, 844),
           new Vector3(654, 0, 819),
           new Vector3(667, 0, 753),
           new Vector3(724, 0, 679),
           new Vector3(732, 0, 646),
           new Vector3(757, 0, 616),
           new Vector3(762, 0, 597),
           new Vector3(752, 0, 573),
           new Vector3(733, 0, 569),
           new Vector3(712, 0, 586),
           new Vector3(671, 0, 595),
        };

    List<Vector3> pequeni2018 = new List<Vector3>
        {
           new Vector3(1840, 0, 820),
           new Vector3(1773, 0, 760),
           new Vector3(1721, 0, 727),
           new Vector3(1648, 0, 656),
           new Vector3(1570, 0, 644),
           new Vector3(1536, 0, 608),
           new Vector3(1518, 0, 635),
           new Vector3(1500, 0, 639),
           new Vector3(1493, 0, 629),
           new Vector3(1490, 0, 596),
           new Vector3(1474, 0, 583),
           new Vector3(1447, 0, 595),
           new Vector3(1398, 0, 567),
           new Vector3(1352, 0, 554),
           new Vector3(1298, 0, 581),
           new Vector3(1244, 0, 567),
           new Vector3(1183, 0, 558),
           new Vector3(1152, 0, 604),
           new Vector3(1101, 0, 639),
           new Vector3(1020, 0, 560),
           new Vector3(1002, 0, 560),
           new Vector3(948, 0, 610),
           new Vector3(845, 0, 585),
           new Vector3(797, 0, 601),
           new Vector3(789, 0, 592),
           new Vector3(799, 0, 556),
           new Vector3(751, 0, 571),
           new Vector3(736, 0, 568),
           new Vector3(706, 0, 588),
           new Vector3(658, 0, 598),
           new Vector3(622, 0, 584),
           new Vector3(561, 0, 587),
           new Vector3(501, 0, 587),
           new Vector3(447, 0, 601),
           new Vector3(433, 0, 555),
           new Vector3(445, 0, 530),
           new Vector3(481, 0, 525),
           new Vector3(491, 0, 506),
           new Vector3(535, 0, 485),
           new Vector3(550, 0, 442),
           new Vector3(550, 0, 388),
           new Vector3(575, 0, 388),
           new Vector3(613, 0, 418),
           new Vector3(624, 0, 403),
           new Vector3(627, 0, 362),
           new Vector3(637, 0, 343),
           new Vector3(627, 0, 324),
           new Vector3(598, 0, 342),
           new Vector3(569, 0, 342),
           new Vector3(579, 0, 308),
           new Vector3(617, 0, 293),
           new Vector3(633, 0, 268),
           new Vector3(629, 0, 257),
           new Vector3(586, 0, 252),
           new Vector3(535, 0, 227),
           new Vector3(535, 0, 248),
           new Vector3(551, 0, 266),
           new Vector3(551, 0, 288),
           new Vector3(529, 0, 295),
           new Vector3(418, 0, 277),
        };

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
        skeleton1997A = new List<List<SkeletonJoint>>();
        skeleton2008A = new List<List<SkeletonJoint>>();
        skeleton2012A = new List<List<SkeletonJoint>>();
        skeleton2018A = new List<List<SkeletonJoint>>();

        skeleton1997B = new List<List<SkeletonJoint>>();
        skeleton2008B = new List<List<SkeletonJoint>>();
        skeleton2012B = new List<List<SkeletonJoint>>();
        skeleton2018B = new List<List<SkeletonJoint>>();


        skeleton1997A.Add(VectorToJoints(chagres1997, 32.93f));
        skeleton1997B.Add(VectorToJoints(pequeni1997, 13.67f));
        skeleton2008A.Add(VectorToJoints(chagres2008, 39.24f));
        skeleton2008B.Add(VectorToJoints(pequeni2008, 15.72f)); 
        skeleton2012A.Add(VectorToJoints(chagres2012, 29.38f));
        skeleton2012B.Add(VectorToJoints(pequeni2012, 12.23f));
        skeleton2018A.Add(VectorToJoints(chagres2018, 26.82f));
        skeleton2018B.Add(VectorToJoints(pequeni2018, 12.18f));
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