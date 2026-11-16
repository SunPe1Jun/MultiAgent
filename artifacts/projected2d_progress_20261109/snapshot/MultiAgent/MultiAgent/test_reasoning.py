"""Behavioral regression checks on existing real VR-TriRef samples."""
import copy
import json
import unittest
from pathlib import Path

from .agents import GazeAgent, HandAgent, ReferentialAgent, SpatialAgent, sample_key
from .context import load_contexts
from .coordinator import weighted_fusion
from .model_adapters import ExistingQwenPredictionIndex
from .baseline import ROOT


class ReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=json.loads((Path(__file__).parent/'runs/pilot20/manifest.json').read_text())
        cls.contexts=load_contexts(manifest)

    def test_hand_is_conditioning_only_and_invariant_to_ray_hits_and_gt(self):
        ctx=copy.deepcopy(self.contexts[0]); m={'mention_id':'m0','candidate_ids':[ctx.candidates[0].candidate_id]}
        agent=HandAgent({sample_key(ctx):{'evidence':[{'mention_id':'m0','candidate_id':m['candidate_ids'][0],
                                                    'support_score':.8,'confidence':.7,'cue_type':'pointing'}]}})
        before=[e.to_dict() for e in agent.run(ctx,m)]
        for _,s in ctx.timed_samples:s.setdefault('handData',{})['rightIndexFingerRayHitPoint']={'x':1e20,'y':-1e20,'z':0}
        ctx.gt['mapped_referents']=['bogus']
        self.assertEqual(before,[])
        self.assertEqual(before,[e.to_dict() for e in agent.run(ctx,m)])
        neutral=HandAgent().run(ctx,m)
        self.assertEqual(neutral,[])

    def test_legacy_modalities_cannot_emit_candidate_evidence(self):
        ctx=self.contexts[0]
        mention={'mention_id':'m0','candidate_ids':[ctx.candidates[0].candidate_id],
                 'relation':'near','anchor_text':''}
        self.assertEqual(GazeAgent().run(ctx,mention),[])
        self.assertEqual(SpatialAgent().run(ctx,mention,[mention]),[])

    def test_independent_mentions_union_and_no_evidence_abstention(self):
        ctx=self.contexts[0]
        names=[c.candidate_id for c in ctx.candidates[:3]]
        mentions=[{'mention_id':'m0','candidate_ids':[names[0]],'count':1},
                  {'mention_id':'m1','candidate_ids':[names[1]],'count':1}]
        pred=weighted_fusion(ctx,[],{'hand':1},mentions)
        self.assertEqual(pred['selected_object_names'],names[:2])
        uncertain=[{'mention_id':'m2','candidate_ids':names,'count':1}]
        self.assertEqual(weighted_fusion(ctx,[],{'hand':1},uncertain)['selected_object_names'],[])

    def test_original_sets_and_rooms_preserved(self):
        index=ExistingQwenPredictionIndex(ROOT/'data').predictions
        self.assertEqual(set(index['scene4_room2/scene4_57']),{'switch','door'})
        self.assertIn('scene4_room1/scene4_57',index)
        self.assertNotEqual(index['scene4_room1/scene4_57'],index['scene4_room2/scene4_57'])
        self.assertGreater(len(index['scene4_room1/scene4_29']),1)

    def test_referential_rejects_monolithic_selections(self):
        ctx=self.contexts[0]
        agent=ReferentialAgent(outputs={sample_key(ctx):{'mentions':[], 'selected_object_names':['person1']}})
        with self.assertRaises(ValueError):agent.run(ctx)


if __name__=='__main__':unittest.main()
