#!/usr/bin/env python3
"""
Bug #0030: jax.jit produces wrong output vs pytorch_eager.

Patterns : [['layout', 'resize_linear_aligncorners'], ['attention', 'matmul_4d_batch'], ['normalization', 'batchnorm_eval'], ['layout', 'pad_reflect_asymmetric'], ['broadcast', 'broadcast_1d_scalar'], ['branch', 'film_conditioning']]
Divergence: rel L2 ≈ ~0.00e+00  (jax.jit vs pytorch_eager)

Dependencies: numpy onnx jax torch onnx2torch
Run: python unique_0030.py
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper as _nh
import torch, onnx2torch

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unique_0030.onnx")
INPUT = np.frombuffer(bytes.fromhex("1f63f9be4cf6f1bd2cea83bff081763e88708ebf9fbf2a3fb4fd9cbe48fce03e186554bfe239d83f2a1f563f90cb0c3f646d0bbf867d57bf5ec877bffd0bf0bdd3df37bfa2a4073f3613013f21a0e9be401dcbbecb082e40a6776cbf0f0d7ebff8fbfebeedb8023d1d338cbe362d12bf4c6c05bfe190703f24fd60bf4574863f4d5609bfed6b213f5e9a9d3fb23bd33ec5b077bd8b349e3f213c18bff793d23ecd2972bf1420923e3474183f33f1a4bffe5c6dbffcca663eb96b7ebfda5b40bf5479913ec34a95bf05cb74bfbf5c1bbe9eaca6be912f1fbe62e96f3fb565b0bfcfbc653f4a33093f4007573ea3c098be7a3821befccef7be2c35bebe1f6cf33fc312aabe102c1f3f23025abf37d81a3f8d4295bfdfa5ae3e2dd8223dac66e2be8838ba3e2c1608bfa0aec2be1ef424bf88cd46bfd3a1a73d7481483f4ad50bbe2e56483f60a0c2be32d85abf180bac3f3c5033be3c33d73f71bf1b3e93320bbfdb60debdd53fe53e1c88643e5a0210be3b28a5bf059e8cbf82c3cb3f7b0ebebf2091113f29460fbfa2141fbd31bb02bf4944e3be18a1a6bfef4bf4be80212140a3d9e9be5c4ea73fef9897bd79fca5be6077b3bf2dc74abf6371bdbee080b2beb9cf39bfdd641ebf0f26733f71a9b5bf0fe1a93ee7fbabbe46d3b7bf272b8e3f7b810bbf417f9f3d6aa44cbd9dac483f521b2fbfb7fa05be3f950740291cb5bf514b3ebf40091dbe1d0b263f21ecbb3f851e093f1ba634bfcf2286be21abe03fffeff3bed53c5b3dc26a72bfdcd220be97336d3e6487063fef0dfa3f001d71be11578d3f97ee1bbe57e62140d1d9033f5fc24cbf7646363e621fa3bff05c6b3e47f4843fd4f788bf53be8bbf08250b3ff330efbe9f21273f565222bfd74a72bfaf2710c07bc2503f1691febe518119c0fc9f1dbfd44c07405ab96fbede269ebe5abaae3e34cc48bfda90edbfb2dcc63fe2bb81be35e71bbfb9c7913f23ce72bf20d1073f0bc1c8be7cc689bfa24aab3e80848fbee74b0e3fb4f7c4bdabb9e2be02a686beb3c9553f6de187bffa42403f93f40d3faf0e003da6d5ddbe778a71bfe8a5e8beb555df3f86a2a43f52f74d3e54a27dbf6c759cbdd6344bbd5f8d44be952acd3fd44b223e064f16bf443cc4bfa1d984bf4cd1573f24809c3fd7eb37be1a15b03ecbd5583f5bb92ebe934de9bef51549bfb542843f38d10cc0ab899f3d9508c33dd77132bf2bc422bf83cd80bf21a1463f19f005bef53badbfcd3833bf6ee0603e14668c3f7612eebdbf2d393fdbea0f3f5537b73fea3d473f445e533e7af6853dc9e1d2be9205d73fff4f98bf9dae433e70b3803fa8dda43f86cdf9becb579bbf88171fc0941a89be81aa013fe51b13bfebc60e3f557e91bee2a6a3befd48293f8aadafbd7ae3babfc6c568bfb99eaf3d7a6f833eaf37223fc0bab83a2d6a6d3f5698003fa2f14b3f02bd10bf4e546fbf6bf132bfe82081bf868d22bf1a66743f52a890be97d3cebe9453853f0cd1523f6d79143eb26c893f0e825bbc4cd07bbfe05f2f3d2ab52bbf13ae993f625b51bf2f23903d234b4fbe304a17be22d9a03e3d680d3fab08b03ec94ca63e32873a3f8a9e7abe009e53bf41d0f4bff7d2393e722b8ebfb809d3bfc7e5643fda7f35bf941b2cbe641dcbbe1ab393be7966193f1ee8b1be0ad4f4be1cab193b7491f1bee9f74640491414bfe485c63e2453b13e3321bfbf5d5d633eac5b61bff19082bf799330bfe41dc33f7ab836bf64f7ecbee08af23df358433fc0a0823fb5b2853f4aa3e53ec823a53ef4d46cbd2fdb42bf5d92f83eba44cdbe53c33ebfcc82a93c0dac833fe26555bf6ce2443ff04464bae941b2bf1606573e9bedc2bf45c1c9bfaeaa2b3fa7816cbd8b1e203ffa4d98bddae8d2be129353bf5ead9b3f76b2973e3789ff3edeb8f13e4e2913bf354b693f68387b3fb70dfabeb9b055bf2fcc3dbfa927a1bfe8eb124078a665bfaad897bea1e5ac3eecab39bf2ed34bbfee8b303ff8f1a9bf240e3b3ff2b7163ffb94c03cc982a13e347424bf14f133bf6e49693f698183be7d3fe9bf22f7193e458a86be0284233f252c8fbfc5702f3e2259873fbf9d4b3f20f755bf5ae1c03f84c5133e7a696bbffa0b813f4304b13ee9cfd4bdf3d58b3e427b0b3e5ea6d7bf610c81be8f83d03fd41ef23fd0ce2ebe35dc1a3fcf0d12bf4e0ef0be0711973e6b3ef93e949c393ff26f0d4012377f3ff4324bbf634ade3f7a82023f3d77503f5b7edebe57855d3f50e1bd3fceaa7abff5a5883ed1c04abfdeaff6be13c5a0bfd7111c3f53028cbe6727e33f9ad5fd3eda1fe1beab705dbf48114c3f2499873dec6e49bfdfd3233d8ab6823f2aed09c05ed6d23ed90812400d7900bf9154de3f48438e3f6c8da13fd27d10bf2ef8e83ef17cd4bfe2e80bbfb2f5513e4f50a43e0b52953e01a9483fcb90f53efabf3b3f99cd463e277177be8296043f75d3893f3e8187bfe7980f3ff6789dbf02af30be0a804a3ff05af2bf55e18c3e4b48afbe0acd7ebffb3badbe5794873faac5c43e9c7befbeb38a4abee97badbf0a6603403d15813fd13f573f93b8813f0ce1a13f48f9743f3015a5bfbeac1dbfbb7861be86d5bcbd2eb9dfbd7ea532bf344c68be985f91beddb71c3e8d2613bd3e578f3f8aa284bf811630bf817c8e3fa012313f8f92743ea462793e680aa2bfb169a53fc43a5b3e76e3bdbe0e083d3f7c47bdbf87d1c53f6e5514bf6f2cfa3f0e505e3fd046c13f96615cbfd76a4fbff105cc3eb05ae4be9f12de3ee54d593fec4cb03e36f325bd4780553f51fbffbdbbad053e686cf2bd6fc10e40b2eaff3de28261bfdb68413f941e37bf9f93283f79d206bff4498ebfb5f391bf1562b8bfae277c3f7e2c84be751070bebb41043eb72b9ebfe4b612407eb43abf4c65b6be01cc91be24c1a3bebbed51bf7bbc2bbfb602543ee1a72b3e18a4b63e33de0abfa2b21f3e19a2d13e9c63b43fe66750bf17fc8a3e0dae0bbfd7803e3f06a7f23e1ccb94be56da7d3f4c34f4bd12c3ab3e5920b93f8f8108c03dc3193ffeefc13de2f6b3bd96091dbf18f5c23f4ae8203f23fbabbfbdff10bfa81c0bbfaa512a3af19ac9bf2518893f644bebbfe81198bf4c4cd13eda3de5be1975aebf0fca643fdd23d3be76b3d43e1d05d7bf532995bf5509e83ec6c0c73fd4dfd0bff763b03e722a2d3b55cd4dbd88a4853fed1d233e18ff39bdd47fed3e0000b93f30d4a33e0e210ec05be02abf667191bf7ca5a7bf667eba3f127532be0b17a4bf44861abfaebaf13fef82513f278db03ffa6634bf97a934bf00a8ce3f52fb8c3f0667a03fb6b7893e5a67ca3f206f75bf5167a03e2c8034be5253eebf8add103ef649a03d840b84bfa189ffbeb101aebff1a5413ecfeba5bf31015d3f1e661b3e981adabe42caea3e1882c53fa6ce2cbe56d9843cc279a03e5c0a79bf276588bf32b2c53e02f9f53fedf6bdbe47c3163ed50786bfd9ce4abd35071fbf704cdf3fca5e883e930083bf382e913f911c1d3ff9505ebf930ab2bf10dbf5bfd0ae39c0a0538cbef5ec833d5c47bf3e94e227c048da15bd9fe8f1bf0095ed3f965be13fe503963f7055a03fec7684bf14e5193e4d066abf31a02a3e8d8ac0bf28efc6bf26eb503f730523bfd59d493f5d873b3e33f7e13fa92b8cbe1ac604401243ba3fd70a33bfd76c7a3f23054cbebd7e633e8fd70e3f12d475bf49a6df3e8568e43e659a9dbe1cb7cf3d1dd7af3c67b37a3d03d0de3cb37263bf3af0cbbd1f298d3ef7237ebf10e9a3bf55866cbf7cb5273fbb0d10bf0241cc3e8359483ff126863f96dd8e3f14a5c43e97e9ad3f441b7dbfa18b9a3f423479bf85c6ef3e48510340d6c2b63e5945783f8bff85bf8f820abd08dc2fbfea201ebf8e3d343e1ebbc5bee899fa3ec8281f3daf6373be407f4b3e6b4e583c7cba4b3e4aa6053f2a19edbf4dc6fbbeb7a9febf630682beff737dbeee10ccbe7644e13d6a629abee40af03e93e6fd3e34cd6c3fcb5726bf3c6651bdf8a987bfba77733d906fbdbf85528fbf15bb193fec0d763f5e9cbcbf84eb86bfef1c4c3faee6243fabd6563f389439be14f87bbff933cdbfc1a01c3f1ad545bd75d2d5bef3a687bf0f8c823f53190f3f2d8cb53e0ee2713fffdf553fe2c629bf835a32bff7127e3fe7bef6be35a2c43fe032e63cf1c8a43f786afe3ec68b1b3fd8f62e3f8d682d3c061aa83ff83f5f3f046bf6be8dc30d3f6c1a9bbe999798be8805c43f0b4dbc3f519a7a3f6bf744bf18f9ae3fa6c1e63cbd1aa7bea8fa313e0635523e7eb1253f691f2cbfc0c955bf9da3febf65781c406cd269bee3cb1dbf54526e3e87b3823fea5368bfe0e4d5bf455bf33ec5054f3fbaeac6bf6dadee3f5399a13e1c34c0be8a3cf23eb0221dbf65bd7a3ee0238cbef518fc3f59f69c3f5752bebe5743bbbef6b1563f34e32a3f5c32c2bf40c270bf2e37d2bff9518dbec52ab1be3bafaa3eafa9573e8392a8bf452e9abda94f213ff60f9b3e1065833fc62590bf8e8e8e3f342264befd4d373f452643bfaa720d3e06db13bfc2624f3da368c0bfa77eabbe53e6dabf1ecdaebf93af28be8278913e92320bbf8a62d6bee278c9be2493403d0cce033eb2adbe3faacee4bf9543e2bdbddd5d4083b68d3fda6f593d4474353f5309673f3435d4be091ac9bf23d1933edb87e9bf5cd6cebd513f52bf7dd3f23e01193bbf9800fd3ec9f147bdd7e88d3dcc1f92bf2147853f6ac1ffbfedfb753fe2c8e6bd6d734cbed719433e62dd4f3d014ad7bca1b685bfcd5a58bf5664b6bfde22673f33d96fbe695f03402d9008be86abf83fdbdf8abf6b401a3ee863f5beb99e7fbdd435b83e7f24a13f3281a03ddbe50d407634213e820730be1a6ababcbeb0863e2d18513f4a66ac3e9d61dbbe884c54bc3faeffbe0b291fc05086f13fa2b84fbf13822d3f19d4e1bfb04efdbec1462e3fb9fe013eebf504406ec148bed44999be4fb4b53e75a438be65ef7ebfba05f8be214f4abf7c00f0bf3caa823f23244d3f299eab3f6e76893f49314ebdb4d192bc8a08853fd3f975bef5cf573f5bf47c3ffbe1c63e6edc09bf886c7c3e9d39b63fb8046bbdfafba33fa4cf033fbf0b2f4063a7c93fe1cb19bf9d43b3bfd7f492bfb5757cbfe7b6fabef170923fc03d0840ff184d3f3e2e1c3f444ad5bf520617401e9fd1be1915103fa6dfd9bed2c985bec144723f9c8274bd877dcb3e82400140801a9e3f4f509ebe63ae4ebfea3005bef7c7003fd01486bfc743c13e253f933da5907dbfe895433f1bde7e3f6f035cbf3dc50a40860f6f3ee83f363ffd9275bf6581b4bd63c1093e85bb94bf913a403f242c92bf8f6c5e3f666b6f3f349858bf62de653f5274efbfd52bc6bf0d1aefbe2673243fef50c93dc821ef3ee5d4583f84a28abf98f86ebf173099bf18f9b3bfb8fb203f5c723ebfab11c83fe5d437be2214983c4bd94ebfd4cbee3f720212c02248a7bf55e4443f61ce60bf9faf643ffbef91bf45b4d5bfb67c41bfb018c13d9187843e8ebac93ef852a9becac41040fc13493f6a4905be0ffad3bf9bc51ebf108e993e30e3c03cb361143fdcf605c0d923f6bd98e8c4bf0d4547bf26050dc0fb117fbe1e06813ea7ba403ece453e3e0d21be3f435cb4bfb88134bfc30aadbeec9f993d23358abf3a6c7fbf86618a3e9acfad3d55100340045bda3e04a7a5bf3e1448be01dcd03fd85e893f68a1153ffe98873faf293f3f9ef49c3e4e0312c0920da43fccefbf3ecea3113f59e22c3fd991adbff0ff15beb8fcd53fec6f08be4b3d66bf6f6156bfb1e303beeca9103e9f921b3f8ab5d6be97963a3ece2b22bfc78e78bdd67c0e3eb7bcd3bf3951933ef68d0440b9d98bbf4168183f41ef963f509516bf25e67b3e4ea51cc090717ebfbe52ad3edd2f10bf8f90efbf51ea5e3e26fe463fe498113f123813bf5ce2bcbd4d71053fe7a708406af52bbe970dcebe12cb13bf76237fbf203ece3fbeec4f3f7a8c7cbea8e5173f95eb3ac08c5971bfa77ee53e06704a3f600e4f3f4122fe3e37382b3e602d7f3f08c54cbfad41983f7e1fe9bec1f502bfc6333ebfad3004406534dd3f3ca61c3c0344cbbfcf7c9d3e9dc7babf58c9623fe1a810bde42cc53ef72d93be655e753dfd5e453f770686bfbeaac73e2a97973fb1d4af3faa1c89bfd2565fbfb8f892bf4a93cd3efe2528bfd5ac15bf356962bfe6a399bf350516bf48d0923fe98684bf85ce0bc03146373f73f6e73ec4b908be92ed80bf5a67283e920e66be8d01d5bed0b1293f66f69e3fb101cdbf4b829d3fcb7bdb3f8654abbdb6cfca3fd56bfe3ec50c4a3e4dabdb3f9034d4bda6590ac00673e4be8a29243f7bb7204007a0893fa63ec73d7d579ebfb9ecf43e38eca5bfd71314bfabc2913faaf3103f4bea5d40160edcbfa0ecb9bee4538b3d8b3b5bbe281d43bf57bf5fbea87cdbbd12d1213df0710d3fab4948bea58ab3bf14acdf3fc1a590bf2be6d53fe224c73e0929903fad23b03e02cae5be78cd44bfe7cb533f8fbbd4be4a35763ea204a13f3e05a0bd910e95bf305e2b3fde79cd3e3896b53dca16104042861a3fb80c46bfb69e3fbf9f64e3be45dc953d79b95a3ee5c25ebe448e21bf31cf0ac08cabef3e568b323f01fe0f40e7a4883e5ed1183ebe3154bf9a45cdbfb036dbbeb4cd0fbf32066bbf101d3abc1164b4bd60a1333f9765453f1c02eebe3fa74d3f18af1fbf5fd6c8becf298b3fc72882bfdbf9473e782cbbbf511c1e3f1a44253f56b5a43f4bb2173f5abdafbe8af4853e4536b1be02f82a3f00459abf57f4acbea00a5c3f150d87bf9f3a893e80a458bf911ababe090f673f3f7935bf1be08b3f412d9b3f292433bfef0e9cbfe01c983fb47b023f9c0b1dbed8b3883d9bee90bdc781acbfcc4088bf7413a73e73c696bf7349113e2e50debd0ea7443f9725003ff6acd63fe5c8cf3e64578fbfa27d05be19010b40de7fef3e19200c3f6cc2babe466b1dbf47419ebfad3e9c3f2fdb8dbcec3528401eb9703f2738ac3e2240eabde56c853f56ed843f948083bee9e3d1beea17ae3d47024ebfe8054940e907953fcc77db3d06b8d7be6864d93e940ff2bb4d2c52be7a9ccf3e5d91d5be7e8e20bdba1f69bf8e6dd93ff495063fbbba973f6b13843f6bea873f5204863f3c45823f1ff1823fe451c73e1f3c303ca4b202409483cebf741956be88d8a03e7a860dbfb1122c40b23d1c3f1670ca3e026461bf8e7fc13f4da9aabf4925dfbe5869573f812091bf254809bc9f570abfed575bbfec56b13c15200c3f3fb1f1bc3c49563f1ceeacbd9422d23ed646a2bea617a43fa505cabf1bee88bd677854bfebf1af3e08a0c63e1411ac3ff7261c3fc424083edcc783bd36a91dbf077d99be45f951bfc180133f9069983f3544bebec8d2ad3eae5d3cbefe45b63e2d5110bfd74667bf07dfdcbf3e011c3fd0f858bfcddfde3e1df368bd6dca463f1d3626be904c33bfd1d1b23ec1164a3f8bb5d2be3b11e63e77fc6ebf8ff6ec3e7711ad3f012e3ebebd86ac3e648cc43e65fcf13f45cf113e41f12540ab5181be35ae473f89d58e3ff7a5313f369d0b3e34fb5b3eddc472bfe7d2263f3ea3533c0798febe1185c63f239e5bbf33eca23f395433bf2e8ea03ff323093fbcd70b3ed6e20840a6522dbf027d553faf3d823dbe93313f70de033fe19137bf3fa58ebff82637bf9d2a4e3e9987a3bfa144f43efb95cc3eb69744bedf1badbef95441bf377e4f3ee6701d3fc6e7473f8c8ca63e3f848fbfd0df90bf8c63dcbfbf13f13f83baa13dcd1fd23e029b5b3f1c69f63eb3280ac04751803cd54816c02c60de3e969d3cbfab0b1e3f37ebd93e351c0e3f6aa3fa3e182ac43f407bba3ed9e7233ee559323ff0497abf04cfadbf9ad03fbe7f239fbf49c264bf398e14bffd4111c01d3308bf6ba87abefa2decbf080d08bfefeb863fef742ebf6cf53bc09e03a13fa2e2b2bf5272963ece56b7bcaede4d3cefeb92bf10fa153f540ca43e4dfa553e5cee5b3fa6fa55beae8a48bfe3838c3f95e9663fca9d7e3e7e3df9bdc0136fbf630f8ebe91b4993f5cb76d3e3750d8beac31103fc1cbb7bf00dfd63fbf1bedbe9a2bee3e70438bbdb43a55bec5c933bfabde663f0835bdbf4eb55fbf4bf23a3e125aef3e240850bfabe09cbd7f90f9be57f380be20c7c13f50b50e3fcf1e4dbf73232abeda7236bf01947b3f65ae1c3fc4e1783d86f0513e96fe6fbfe81ed63e3298953f83d99f3ec1aab6bf6d4d56bf6721683c6bd3453f89e3c3bdb577033fec369f3ed37f14bf8a6c863f55c6903fcc334dbd880cfdbe799e53bf923b14bf28ef913d93d6573e6df1aa3f0f78d1bf8c3369bf21d0413f38ee6c3f4c01373fc77db3bfefe9e1bf9e1b1540c9f511407c273bbeae41df3efd141ebf4e458b3ef9d8113f654aee3d1167abbfb2cb3bbfa2dfe23e2f88e43e5d1dd4bfaece893cf20d7b3e83835ebf121280beca0afcbea540b7bf8a9d0e3c1576e53eeb52ebbe006a0e3f5cd1ad3f72ec0ebfb6b5cabee7b6ccbf1a080a40f93f044076f3203daa787b3f82031bbf77bc563fb243a1bfddee663f4ac4dbbe441fc73f0cbc9dbfb8f82bbfdf0f45bf075c65bf76ef0cbf1a5f473eb71d1b3f20c41ebe7f2c30bfec1c3a40ed2538bfcde053bf6a2b80be9976663fb1b899bfe99f3a3fdc8f443ddb1edebd0831054092387dbf14cab6beeafda43daadde63df76b87bfd076efbdc585883f9fac5f3fc7ecaabe783876bfaaaea1bf5f2fb4bf675691bc96b5fc3d85b98cbfe4e830bf6e002dbf75871b3f4c2c76bfb94621c05336a9bfa1fab5bf0728fcbd87e286bf62b5fcbd07d0513f8996a63f20d13e3b68c447bd58528dbedfb4623f74dcd73e7add7e3ea095033f0f3aed3ebbd98d3e9f60083fd69d563e6d8bb7bf67a9b6befaefc13fd207103f82b9efbf7774b83f1688713ec8133b3fc90bf03e769b15bff2fe1e3e8ec4503f53d70cbd9c1c7fbfa40a143fc28e38be2ad0a0bfea2690be6b7fe73d7af7fb3e1582c9be0eff0dbf6313f4bfe3caf93e48bf0140b5f2fb3eb24315c0d429cd3e4b964c3fc49788bec26a1040c44a39bf6066703fe291c3be9e9c983f0dac953e573ad8bfa6dbf9bec173293c1112ff3f291a69bd2e1c58bf569eae3d8090783fe7ca08bf56d3c73fc26d86bf0ff89abe416304c0202c383fe1ab4e3e04dafd3f9a867d3f30d6a13f774427bf3e6fb2bf9af9c63ebfeb8dbd767ca6bfa10d083fbae62e40f036ab3b64d689bd09a015be832c83be4d8a0ebf80d015c067bf1c3ec15d13bf5e1f893f909d13be131dcebfa0eec9bf269ad7beaf2dd13f92e616bf0f851dbee0b84ebf475aa13f6a364fbf1e20aa3ebfdab93f8894913ffe417dbfe0d3993e4bd543bfbe9e1e3e648da8bedc4fc1bf103ae6bedf20b7be1355833fb00301bf6d9876bea4b7963f3fc4113f664e293eeac4acbec6138bbf79c38bbefdd4cebec519f0bea2707fbd36d2383fba13553f766360bff0749dbf6484fa3fd9e06abf4e9b153f0590403f3365a73efcbbca3ff8759a3c515a4e3e1ac0a83e10b5983f57eb71bf09ae8cbec0f406c01545683fd27d28be16864abf75df363f0e63663f22512cbf04348bbf8f98163f583820bf390135bf449e1340053b69bfcb7d07bf4749a7bf13d65f3f43b98c3e8165d7bfde4504bfce527bbf89011e408bc2993f44c0873fca10a93f6f5be3bfeeb82a3fee72833f9d610bbf3a673d3f2876c03ecae0483f7bfec53e6d7f87be8b2ef53f8128a3bf1f04dc3f0d3ae2bf85e2babe17ee853fb86700c0f2d9c63fe4cf12c041db553efb00ca3e2aa18fbf7ef498bfff17cbbd6d88ae3f21f81abf19531dbfa96ea33f283bb73e3c677bbf92d2373fb6650c3f9e98be3f81710dc00821483ffa40cbbf81406d3f37b363befe0b143feaf2bbbf6be8913e653c3bbfc64218c08e39393f6d3353be503a7a3fb3d35fbf79f79d3f6e0c0dbf32366dbd8b0cfdbe43cced3ee68abd3f13b49cbf907e8dbece51da3d9022263f698623bfb25927bc88a6093db849513fda89d6be0091a1bee9ecfcbd026de4bfcdc432bf6f9ae73fc1c4a5be30b60fc0848bf2be0ae6ff3f21df9abf00ca84be15aacd3f0b4e9b3fbdf263bedd3d44bffc062f3d3076b43dc3a5d03e6c938e3f4295a7bfc77a733f8f87c43fd9fcb23fd020893f7952323fb661803e9dbd52be3642d8bdf917dcbd7b7854bf762e2d3c3a44943f197da83e14dcd7bf24bafa3e257712bff27d0dbe0980b63f4bcb053f8260513e388594bf90b87e3f9a722c3f64837a3fb078743fb4ca8bbfbe5decbe1c32dbbec18d1dbf52708b3f71ef313f20748d3f3308c2be151741bf0f43ccbe6d2cabbfbc3a0540122b813f195d8e3e5e23b0bf063ea7bfbb3334c077e9043f04268ebfa2fd1ac004d5243e7f9c5c3e52128bbf3f2779bf25cdb63ed49a71bf6efa5f3fce1efcbfb31d8fbe5729a73fe51fdcbbbc0bb7bffdf3cf3e296217bfd657873ffe07393f61f1cb3e21ba5abf8d5a253fefcf603f1a8606bf893c383de50d66bde2ea9b3edb5547bf2560b93e256b34bfc587283f554e383f425749be203c14bf4f6ea83e8deb573fcd1e023ec2ab783e0bff6a3f4900703fce758cbf17152b3f4c2bab3d206681be880b363f7d37af3f9ea5b13d18cfa33e877083be26618b3f1f4dadbf730ba23f729e133db5a459bf086786bfa036a6bfcdcc09bffa07bebf7b0d8b3fc868c23f9fa321c07587a7bf7e72423e399625bf7e375f3f61bddcbfe69830bf9d1faf3efa8220bfdca116bfdff1e1bf22c99f3e228df23ffdce743f7764a93eaeea10400612113fd082aabff7f2b9bf2c97973f3c3982bf8acba6bf365e983e028ba9bfece0363f08638dbe821367bf0188803ea671e3bdd57a833f3ef9174018941bbfee33943ff5f99abee465ffbd8ebfb0bf1ca53f3e34ebfabd6fcffa3fb2d99bbdd761cc3feff7bc3f085e57be2f20183efecb173d332f553e69f0a7bea08e3bbef6ebd0be9b51d63d6f14833fe3a8903eb1e3f93e5cb1733e7add0dbfb62d303fc938c6bc510940bf6fee233f4d85be3e4f3f91bf9020a4be52cff1bec7aa383f54b7dbbff2c99abf8955543e0336fd3dc064893ea9df9b3fa9c410bf24d43d3f00274ebf869b8abe8a6ce73e9ef18abf865126c098a6a43febbae53e84167abe60868ebff1829abffc7085bfcec2d23f4479c1be0b0006bf598b263f84b8c33fe913963f81abfc3e88eee03fab81e73c9ce202bc57b5583f3461e63e1204bbbf517fbcbf760d52bff5d958bf66f222bd638508c07380a1bfecf5783fe5fd873f42b0863f9be93640819c23400547a13f587c293efff0073ee26d9abed657943e9269d7bf9ab3873f83855cbfc670fa3ee12d113df067c8bf312394bd5a7d9a3f5f4ede3e43ea623f5e94973f7c59ba3ea7e8273eda390d3f3341ce3efc67a63eb02d4c3f00c598bf7b1e42bf631a8b3fa4aca33fce20113fed8c153eac9d323fce0a2dbd601576bcf4111ebfa70becbe9e78973fde5d283e9a95633f848b693f451bfdbf9c2b2abf9a56ec3fe11c13be942ac8be980614bf4ca4c93e6681f0bca58d4b3fb0e9c5bfeb0c4cbf81bd6f3f1e482abf4c9e9f3f0505adbf8e401dbf121496bf1d3cafbf0d74b1bdf60e123f0a5a953f6999943dc760dd3e3df8afbeb9230c3ec164d1be73d9a13d4cc805bfa546fcbfba04f03e59da8abf80c5153fce2f33bf8326663ff3f22b3e19da61bef7c6acbebfe610bf578df23fc5ab773f608a0c40c7d2cf3e7f8019bf9e76aebe291900bfbb9c863f0e3dd5bf4bbec53e128d883ff95d88bff203a4bec66831bf97410a403b68a63ef5f7c2bf94728b3f130930408f5988be34961fbef1c282bedbcbaabc91250740f25bee3ddcd0213ec7ae1d3de35c09c09a7c46bfdf870abe71932e3eed675b3f57cf26be1c78dd3e0595493f7f48603e95cc453fde5283be82c8983e4e493cbe1c6b50bf798071bfda238b3fa7e95b3dafc0893eaa52363f68d8b63ff3d48d3ed9d8b23d8e8715bf8d46ab3f8fdf5e3f507b74bec66903bf6c5a69bf77e767bfb00fffbe91db0ebf8558153f2b50203fb8c944bfa15f653ea14157bf5e235bbe8bcc18bf1283c93e9862643f92e0713f80a6333c6e9673bf7c1cb3bf9d14893e4aabfc3ebc3f0abe83bb48bf63b66abf754c7b3f62e9f83f48220ac0ee1cac3dac56f53fe287cb3e85d5ad3ea443d0bdab5085be3f415bbe4d499dbf654330be9f13b3bed71452bf30ec6ebf43e308bfe58cf43eae3a673f4722a83f767281bcf23d933fa60c83bdcb83033f44dc1ebf27976e3f8f4920bfeab6b83e06dad2bee4b01dbe0464b63eab79193f9f5f293e8ed8a83e32c8e0bf9854aabb55f79dbf968d183f6cf00abf37acebbe94f009c0cd5672bf86905abc2a981240982982bfc324dcbe3f9e36bf6ad8143df7cf4e3f62b6e63e131d993f3c4ce33ef938c13fdfcd633ed69acdbdcd1fab3fb1c62a3fe397443f6af5f53e5cea84be5dd751bfee5d6a3ef5b51a3f80b4b63e1428a93ed0e63c3f845c49befe31af3d31d8c33ebc91ec3fc3f6cebf008642bf3be504be607e423f64870e3ed2a6bebf1d84853f5eb6d43da90453be7a73e43d1a5f4cbff3e064bfa64106bf6f0d5d3f76a4c5bf431b05bfcee322bfef9c1bbf1dc150bf19a082bf7e6933bfc93c263fd476eb3eacfd1bbf2e7d333ffc1ea5bf5bb9c33f4d1abdbf668031bec8953ebf76d96f3d8128843fa0f50c3fe51f98be089874bf9972f1bedd3489be8eb020bfed04fabe4159953f40836abd406ababf94896cbf821a133eecd395bf835a0fbe368f44be489f5c3ebcf09a3f67e58dbeb3df5fbd18268f3eda460d3fc9ea83bf7389b23f838ebcbec6e5b0bf9f3b343eee4f9e3f9b84b93dfd36f93e7896c53f9c534cbe9021e13cfc59ae3e4484013f380b663ed7cf023f398adb3f473c533f706f373f7b7784bf860210bfca5f063e3194fd3e34e0293f4612abbfe2bdc93e7752e03fa5d2e2bdf90b1bbf57b7a5bfeac2d2bf79e87abf28746ebe1bdacebd3249dc3f0fe61fbf217b4cbe8ce6f8be945b4fbf998e5f3e9d36693ec4da63bf89ee27c02cbd303faca4ad3edade1ebeacda9e3fbd6d14409bb7553e25f879be798ba8be31ef35bec9717cbfddce803ef129b03f84750b3d3a78c23dd701683f6fe4ddbfb7caaa3fb81db73f775941bfb129733eef8093be92c5ff3f7911b23f2279b3bfa4fbb83ce691963f686005bf18720dbfd8f57b3e0a90ff3ed11fb1bf7500d63ec0441c3fd9e3a3bf8ca766be1746c8bf4140833f8474933e820014bf56415fbf89581ebe84569fbf5484053f970894bf993d82bf960bcfbfe0fe333fb7f6413fe505293f441188bb4ab5513f07bbe4bceca9e93e1cbe9f3f1ac28dbf808ac2bee4d8b83f760af9be214cbb3e032496bff3209d3f502954be3a58f93e9b1990bfadec00c00e7e5f3f90a724bf66a9eebffb199cbf48d649bf75bb42bf68c94a3f277ab3be652978bf72f3d03e0bb008beb12389bf9f8bfa3f08fcb73fab23903fe1cdf63f074bfebe4a8416bfe1beaebfac7b3c3e91f8a73f4b12b7bc39469cbf49fd87bf549cc03e2af8563fa7fd5c3f814ac63fd380d53e8e07ca3f57d68e3fe5e483bf201b3b408086623f2333bfbf7bb4eabe06ba46bfa95c02be1680453d1f2b04bfbba7373f63fe323fd584e43e7a21c13f312d7a3e184c6f3e01c8ac3ffb5f55bf841968be112553bf0854f53f99d181bf225d2d3fb22006c009e2ac3f7b4a5dbff4e292be0c791b3e50ae60be1c3ad7be7295b23f1f5d973f78438fbfbb3b073e99f0024029c0673e8801d23e0b5a6c3fb08241bff05badbe0d4cae3f599f2cbf88cac6bf546ca13f28125f3fd2e42abf0af5e33ead614cbf2cb3bc3f114bb6bf9d25103f702a54bf0b9e2abe369074bf7224e93e96430a3ee2a300bfe4eae3be1e5c11bfae3516c0966dd73f038307bddd4f83bee4dba13efb7db33f3593dabecc18a5bd1732e93fa5d0eabddf35ffbf39581cbe3927f23f4f5a483f8d3c3abf7a94b2be7bd8a53f48fdda3ec6ea5e3e2d14933f83a7d2be4d65d23e60ed323e2d10cf3fbd0ca9bfacab33bf38a4353fde9ce8bf96e49abf910c93be53a6ce3e3df18bbfa684f7bcbce4b53f45e13fbf8647a7bfdbea77bfa8cc6dbea467933d03a923bf2167f8bee5463b3e03951f3f91f3fdbee56ac8bf6d34183f9eb7693d208adbbd7e4f593f214c6fbec8cae2bf34e0e8bdcdcd8b3f334d123faf38a1be3437e53d3775d6be20fa57bda93309c0a0ac8a3fe46ac53f67ad1b3e442ebb3e242a3e3e2c813940b5bc1e3fdc5676bedbdc5fbea412233e9377d13f8951c73fd373213e486f9abe3c400d3fae5eba3ff7c5463f08ea533f27c52bbf0e19b63f89cfbbbe3c04093fa724173f773a853e693291bf76717abf1b9764bfab6fb03f28bb13bf6059483fa1a458bfdccd773f54b3133f060f4c3f500e893e395c78bc204e80bf92280340ea03b3bd5a51d0bfe7600340a939ffbf5ca3ad3e9ccdd6bd4f3c933ebf2672bffdef8bbe1427033f8db225bf7552a8bf4e946e3f96b5cf3edb33ce3fa6623cbfcb2da13e8be8c23f590381bff208603f19b7133f58f713bf4e9f1b3f58630bc0934667bf8ab7b23e582e14bd6b0465bfac318bbe9d390440f529093d2518eabea0c825be69037e3f1983cd3ec2b8de3ff7323fbf0e860f3f84454cbff6868a3f8381813fe3168abe6b6a4a3f02e2c13e245fe73f4a30a83edc4796bfa610e9bd6975d5bf698eb5be0085b53e20a41a3f98457fbe523c8ebfd193323feddc0f3eb586f83ecbbe2bbeb31faf3f83bbb23fa4ac373fee35eb3f2db32ec03083a33fdde51d3f376d04bffae78c3f453ca83e18335ebf55f481bf30cfcebf237a1d3f0ef75c3f6d59413f84518d3ca7790cc0c6d9f83f101d79bd9daaba3fa22fa7bfed5094be6f3d733f22ae963da549373fd4467bbedaf4993e343fccbea37adebd4c54cbbfd6d4dcbf8c5794bef7b68b3f116494bfe7eff03deef5813f015ca43c0001553e5c613f3f588ade3eacb24b3f69a96b3f269197bd1dfcf93ddbbbfbbe62d2253f395e2e3f2d64243fedeababe021948bfbe0d23bed4705fbf0300163f047cb9bddf6a0ec0868f4cbf28f6293f23808b3fe0070540cbcc763f26941abfe4980f3e6afd823eed46903fbe6ba63fbf0fa03ee76851be371f933f12ce3bbf1afc27bf4428da3f02dca4bdd1e0ca3e41d2e7bf1699b13e8dfd983ff4251b3fed2c243f5a9c7a3fc529cbbe36929f3fd27c40bf7722cdbf59020b3ea0b39c3e7e228f3d844acc3e808195bf183d08c0109f293fe3ec763f276891bf4512403f86870b3f677943bf31dc973fd3c0b5bfb64bf33e25e505bea84c8ebfce3501be6d40333f14ab823e10e60c3f24098a3f6690583feabd6e3e938d653ff7addb3ecbcdb63f64211cbe7488a73feac70ec0ef6a57be8911cfbff5e8d83ec061d03eac7562bf3a6ca4be5278edbff2c7253f4ec668bf456baa3f70741fbf8384d53f2216803fc2ccb33e537242bd0b31863f54b6dfbf3e5a93bdb89f553f1947a13ffdaab43fef80ba3ef4ec993f2bd0d93f781bbd3ea3329dbfeee932be1d162c3f48dfb63f6bdaaabd973414bf5633043f7551e73e237d35bfc52f263fd262fabf26993dbf9048cebde52e19404186c5bf068170bed2535bbe47a75d3f136605c0fe2daebe83a7f13e524a7c3ff1898e3f51babdbffcff2e3fb5743a40d4348cbf70f31b3fd082d8bd4e3a37bf97bb0dbf8592c0bea9098e3f10074bbada42bfbf512e27bff02ab3be00446ebfd1c9f93de7cba33f66ec7d3f7395a0beb5b7f9be45d4453f81abeebe458f97bf74c2e7bebc136dbf4311da3e56640b3fb83daabf2b6a183fba5ab73f8a4623bfa22fa4bf3c8aa73f510e80bf1337cabe55fb4bbf8a77a8bf1e094dbe827d9bbf0e0fd53ee66ce53f15e0ffbd715d2b3f1392853ec6f78f3ef9fb7b3f2baf5b3f842c1c3f3983d7be8f07d0be29ee37bff1989d3fc482b33e00866a3fae0dabbe5df76cbf2ef3963d61fd7e3fbccfb83e3a8c0ebfe6b9dcbe7323073e292fa9bf8033dcbdb505f7be0e08a03f679d95be5a7c39be0b6395bb0db3c43f5e0d77bfb22afe3e82f5b13ea824743c54615fbb7958eabdfd9b503fdf7ab0bf27bc86bf8294c63f98aa843ea6410ec0125b333f25f2f1bfe335b2bef454bdbe2a0b6cbf9a9605bfe51599be0e61b53fc7eff03e1754853ff54e25bffe934b3f0759823f6000143f00d4563f59a61bc00733703f690978bffbd297bf513508bf58fcafbec64da03ed4c885bf39ca1b3f3738abbe8692a3bed08fe9be0394e0bfd6cda5be82026dbf266093bdab8dcd3f2f53c93fddfa393ff55181be7c3d62bf9af4c6bee856df3d06f6e3be915a16401c9c833f33e7cebd5aefc73f64d8553f2f276bbfd503dbbe165dc3bee6a093beb4d6fb3ea27ece3fb032ab3e3adebbbfa4244abf2ba2493ebafb8ebe7bba3ebf677b693f0a200bbf2f56a0bd056106bf575471be85a7e63e0cb374bf82e0f4bf9b99053f7aab2bbfd35c693f1740763f1409313f83ed85bf61fdbd3e2bb3c43ff5fd07c0bc1b9cbd0dde9ebe6d3e0ebf42a80740146da03f6dabe93cddf273bf2d4f2dbfb01dffbca53f9f3e1f18abbfd610c4bd002813c09bab7dbe98f789bfabe3033fca7fadbd5c9fcbbff4b216bf95db3a3fac51a53d9d3e353fa750cd3f9f19413edbfba23e765f593ff1a654bf6007f4bed522fe3e5dffd2bfc140553f937b143e3ee8db3eacfcab3f3efc3ec050a141bea37a013fe7347bbfef5fcf3f0aa606bf435a1abf195cc7bef1bd853fad648e3f585838bf724d4f3f953cdfbfc294e83d25b56b3ff22543c088ba743f82328b3ea72418c013a303bf40cfdb3c9043bd3e68df36bf3f731a3ef05d333f6b5bda3d37e7a3bfbd38dabe4d6d8f3eef5980be7e8e2b3f7024b93edea7273f5077193ff1eaf13e392210bf641299bf893471bf855257bfc49ae33e0b6e8fbf7f71f7bed5331ec0e06bd0be5ce1013efeaf60bfdadaf0beda738d3f4a5c564000029ebcc0f9253fe51a3c3f12709b3f787cfcbd36c2df3e9aad153f964aee3e8967e4bd143f503fe284633dfc4d8d3f8716803e404b253fb095923ed8a72ebf7c9d243e7b882fbf32b57c3f9cfee43e70569fba2441223f5c875b3edbb3a1bf6ae502c09b797b3fc0c7f63e2559024007e2a03e86a4ce3f50258f3f320213c0"), dtype=np.float32).reshape([1, 3, 32, 32])


def reference():
    """pytorch eager — ground truth."""
    m = onnx2torch.convert(onnx.load(MODEL)).eval()
    with torch.no_grad():
        return m(torch.from_numpy(INPUT)).numpy().ravel()


def target():
    """jax.jit — under test."""
    import jax, jax.numpy as jnp
    model = onnx.load(MODEL)
    inp_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    inits = {i.name: _nh.to_array(i).copy() for i in model.graph.initializer}

    def fn(x):
        vals = dict(inits); vals[inp_name] = x
        for node in model.graph.node:
            for nm, v in zip(node.output, dispatch_op(node, vals, jnp)):
                if nm: vals[nm] = v
        return jnp.asarray(vals[out_name], dtype=jnp.float32)

    return np.array(jax.jit(fn)(jnp.array(INPUT)), dtype=np.float32).ravel()


if __name__ == "__main__":
    ref = reference()
    out = target()
    diff = float(np.linalg.norm(ref.astype(np.float64) - out.astype(np.float64))
                 / (np.linalg.norm(ref.astype(np.float64)) + 1e-8))
    print(f"expected (pytorch_eager): {ref[:6]}")
    print(f"actual   (jax.jit):       {out[:6]}")
    print(f"rel L2: {diff:.4e}")
    if diff > 0.001:
        print("BUG REPRODUCED")
        sys.exit(0)
    sys.exit(1)


# ── ONNX op dispatcher (required to run the model in JAX) ────────────────────
"""
Shared ONNX op dispatcher for JAX, TensorFlow, and other array-API backends.

dispatch_op(node, values, np_like) executes one ONNX node using the provided
numpy-compatible module (jax.numpy, tensorflow, numpy, etc.).

Rules:
  - Initializers are stored as plain numpy arrays in `values`.
    Shape-extracting ops (Reshape, Slice, etc.) call np.array() on them safely.
  - Intermediate computed tensors are framework arrays (JAX/TF traced).
    They are NEVER passed to np.array() — only used in framework ops.
  - The dispatcher is framework-agnostic: pass jnp for JAX, tf for TF, etc.
"""
import numpy as np
import onnx
from onnx import TensorProto

_ONNX_DTYPE = {
    TensorProto.FLOAT:  np.float32,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT32:  np.int32,
    TensorProto.INT64:  np.int64,
    TensorProto.BOOL:   np.bool_,
    TensorProto.UINT8:  np.uint8,
    TensorProto.INT8:   np.int8,
}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:  return a.f
            if a.type == onnx.AttributeProto.INT:    return a.i
            if a.type == onnx.AttributeProto.STRING: return a.s
            if a.type == onnx.AttributeProto.FLOATS: return list(a.floats)
            if a.type == onnx.AttributeProto.INTS:   return list(a.ints)
            if a.type == onnx.AttributeProto.TENSOR:
                from onnx import numpy_helper
                return numpy_helper.to_array(a.t)
    return default


def _np(v):
    """Convert a value (numpy or framework tensor) to numpy. Only for initializers."""
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def dispatch_op(node, values: dict, F) -> list:
    """
    Execute one ONNX node.
    F  = framework module (jax.numpy, tf, numpy, …)
    values = name → tensor (numpy for initializers, framework array for computed)
    Returns list of output tensors.
    """
    op = node.op_type

    def get(i):
        if i >= len(node.input) or not node.input[i]:
            return None
        return values.get(node.input[i])

    # ── Element-wise arithmetic ──────────────────────────────────────────────
    if op == "Add":       return [F.add(get(0), get(1)) if hasattr(F,'add') else get(0)+get(1)]
    if op == "Sub":       return [get(0) - get(1)]
    if op == "Mul":       return [get(0) * get(1)]
    if op == "Div":       return [get(0) / get(1)]
    if op == "Neg":       return [-get(0)]
    if op == "Abs":       return [F.abs(get(0))]
    if op == "Sqrt":      return [F.sqrt(get(0))]
    if op == "Exp":       return [F.exp(get(0))]
    if op == "Log":       return [F.log(get(0))]
    if op == "Tanh":      return [F.tanh(get(0))]
    if op == "Reciprocal": return [1.0 / get(0)]

    if op == "Pow":
        return [get(0) ** _np(get(1)).flat[0] if isinstance(get(1), np.ndarray)
                else get(0) ** get(1)]

    if op == "Erf":
        if _is_jax_module(F):
            import jax.scipy.special as jss
            return [jss.erf(get(0))]
        else:
            import tensorflow as tf
            return [tf.math.erf(get(0))]

    if op == "Sin":   return [F.sin(get(0))]
    if op == "Cos":   return [F.cos(get(0))]
    if op == "Floor": return [F.floor(get(0))]
    if op == "Ceil":  return [F.ceil(get(0))]
    if op == "Round": return [F.round(get(0))]
    if op == "Sign":  return [F.sign(get(0))]

    # Element-wise Max/Min (binary)
    if op == "Max":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.maximum(result, t)
        return [result]
    if op == "Min":
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        result = tensors[0]
        for t in tensors[1:]:
            result = F.minimum(result, t)
        return [result]

    # ── Activations ──────────────────────────────────────────────────────────
    if op == "Relu":
        return [F.maximum(get(0), F.zeros_like(get(0))) if hasattr(F, 'zeros_like')
                else F.maximum(get(0), 0.0)]

    if op == "LeakyRelu":
        alpha = _attr(node, "alpha", 0.01)
        x = get(0)
        zero = np.float32(0.0)
        return [F.where(x >= zero, x, np.float32(alpha) * x)]

    if op == "Elu":
        alpha = float(_attr(node, "alpha", 1.0))
        x = get(0)
        return [F.where(x >= np.float32(0.0), x, np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "Selu":
        alpha = float(_attr(node, "alpha", 1.6732632423543772))
        gamma = float(_attr(node, "gamma", 1.0507009873554805))
        x = get(0)
        return [np.float32(gamma) * F.where(x >= np.float32(0.0), x,
                np.float32(alpha) * (F.exp(x) - np.float32(1.0)))]

    if op == "HardSigmoid":
        alpha = float(_attr(node, "alpha", 0.2))
        beta  = float(_attr(node, "beta",  0.5))
        x = get(0)
        return [F.clip(np.float32(alpha) * x + np.float32(beta), np.float32(0.0), np.float32(1.0))]

    if op == "HardSwish":
        x = get(0)
        return [x * F.clip(x / np.float32(6.0) + np.float32(0.5), np.float32(0.0), np.float32(1.0))]

    if op == "Mish":
        x = get(0)
        return [x * F.tanh(F.log(np.float32(1.0) + F.exp(x)))]

    if op == "Sigmoid":
        x = get(0)
        return [np.float32(1.0) / (np.float32(1.0) + F.exp(-x))]

    if op == "Softmax":
        axis = int(_attr(node, "axis", -1))
        x = get(0)
        x_max = F.max(x, axis=axis, keepdims=True)
        e = F.exp(x - x_max)
        return [e / F.sum(e, axis=axis, keepdims=True)]

    if op == "Softplus":
        return [F.log(np.float32(1.0) + F.exp(get(0)))]

    if op == "Clip":
        x = get(0)
        mn = get(1); mx = get(2)
        if mn is not None:
            x = F.maximum(x, F.asarray(mn, dtype=x.dtype) if hasattr(F,'asarray') else mn)
        if mx is not None:
            x = F.minimum(x, F.asarray(mx, dtype=x.dtype) if hasattr(F,'asarray') else mx)
        return [x]

    if op in ("Identity", "Dropout"):
        return [get(0)]

    if op == "Cast":
        to   = _attr(node, "to", TensorProto.FLOAT)
        dtype = _ONNX_DTYPE.get(int(to), np.float32)
        return [get(0).astype(dtype)]

    # ── Shape ops ────────────────────────────────────────────────────────────
    if op == "Transpose":
        perm = _attr(node, "perm", None)
        x = get(0)
        if perm is None:
            perm = list(range(len(x.shape)))[::-1]
        return [F.transpose(x, perm)]

    if op == "Reshape":
        x = get(0)
        shape_raw = _np(get(1)).tolist()          # always an initializer → numpy safe
        orig = x.shape
        shape = [int(orig[i]) if shape_raw[i] == 0 else int(shape_raw[i])
                 for i in range(len(shape_raw))]
        return [F.reshape(x, shape)]

    if op == "Flatten":
        axis = int(_attr(node, "axis", 1))
        x = get(0)
        left  = int(np.prod(x.shape[:axis]))
        right = int(np.prod(x.shape[axis:]))
        return [F.reshape(x, [left, right])]

    if op == "Unsqueeze":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            axes = _np(get(1)).tolist()
        for ax in sorted([int(a) for a in axes]):
            x = F.expand_dims(x, axis=ax)
        return [x]

    if op == "Squeeze":
        x = get(0)
        axes_t = get(1)
        axes = _attr(node, "axes", None)
        if axes is None and axes_t is not None:
            axes = _np(axes_t).tolist()
        if axes:
            for ax in sorted([int(a) for a in axes], reverse=True):
                x = F.squeeze(x, axis=ax)
        else:
            x = F.squeeze(x)
        return [x]

    if op == "Expand":
        x = get(0)
        shape = _np(get(1)).tolist()
        return [F.broadcast_to(x, shape)]

    if op == "Gather":
        x   = get(0)
        idx = _np(get(1))          # indices always come from initializers
        axis = int(_attr(node, "axis", 0))
        if _is_jax_module(F):
            return [F.take(x, idx.astype(np.int32), axis=axis)]
        else:
            import tensorflow as tf
            return [tf.gather(x, idx.astype(np.int32), axis=axis)]

    if op == "Concat":
        axis = int(_attr(node, "axis", 0))
        tensors = [get(i) for i in range(len(node.input)) if node.input[i]]
        return [F.concatenate(tensors, axis=axis)]

    if op == "Split":
        x = get(0)
        axis = int(_attr(node, "axis", 0))
        split_t = get(1)
        sizes = _attr(node, "split", None)
        if sizes is None and split_t is not None:
            sizes = _np(split_t).tolist()
        if sizes is None:
            n = len([o for o in node.output if o])
            sizes = [x.shape[axis] // n] * n
        sizes_int = [int(s) for s in sizes]
        indices = np.cumsum(sizes_int[:-1]).tolist()
        # jax.numpy.split uses indices; tf.split uses sizes
        if _is_jax_module(F):
            parts = F.split(x, [int(i) for i in indices], axis=axis)
        else:
            import tensorflow as tf
            parts = tf.split(x, sizes_int, axis=axis)
        return list(parts)

    if op == "Slice":
        x = get(0)
        starts  = _np(get(1)).tolist()
        ends    = _np(get(2)).tolist()
        axes_t  = get(3); steps_t = get(4)
        axes  = _np(axes_t).tolist() if axes_t is not None else list(range(len(starts)))
        steps = _np(steps_t).tolist() if steps_t is not None else [1]*len(starts)
        slices = [slice(None)] * len(x.shape)
        for ax, s, e, st in zip(axes, starts, ends, steps):
            ax = int(ax) % len(x.shape)
            slices[ax] = slice(int(s), int(e) if abs(int(e)) < 2**30 else None, int(st))
        return [x[tuple(slices)]]

    if op == "Pad":
        x = get(0)
        pads_t = get(1)
        pads = _attr(node, "pads", None)
        if pads is None:
            pads = _np(pads_t).tolist()
        mode = _attr(node, "mode", b"constant")
        if isinstance(mode, bytes): mode = mode.decode()
        n = len(x.shape)
        pad_width = [(int(pads[i]), int(pads[i+n])) for i in range(n)]
        if _is_jax_module(F):
            import jax.numpy as jnp
            return [jnp.pad(x, pad_width, mode=mode if mode != "constant" else "constant")]
        else:
            import tensorflow as tf
            paddings = tf.constant(pad_width, dtype=tf.int32)
            return [tf.pad(x, paddings)]

    if op == "Tile":
        x = get(0)
        reps = _np(get(1)).tolist()
        return [F.tile(x, [int(r) for r in reps])]

    # ── Linear algebra ───────────────────────────────────────────────────────
    if op == "MatMul":
        return [F.matmul(get(0), get(1))]

    if op == "Gemm":
        A = get(0); B = get(1); C = get(2)
        alpha = float(_attr(node, "alpha", 1.0))
        beta  = float(_attr(node, "beta",  1.0))
        if _attr(node, "transA", 0): A = F.swapaxes(A, -1, -2) if hasattr(F,'swapaxes') else F.transpose(A, list(range(len(A.shape)-2))+[-1,-2])
        if _attr(node, "transB", 0): B = F.swapaxes(B, -1, -2) if hasattr(F,'swapaxes') else F.transpose(B, list(range(len(B.shape)-2))+[-1,-2])
        result = np.float32(alpha) * F.matmul(A, B)
        if C is not None:
            result = result + np.float32(beta) * C
        return [result]

    # ── Convolution ──────────────────────────────────────────────────────────
    if op == "Conv":
        return _conv(node, get, F)

    if op == "ConvTranspose":
        return _conv_transpose(node, get, F)

    # ── Normalization ────────────────────────────────────────────────────────
    if op == "BatchNormalization":
        x = get(0); scale = get(1); B_ = get(2); mean = get(3); var = get(4)
        eps = float(_attr(node, "epsilon", 1e-5))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        x_n = (x - F.reshape(mean, bc)) / F.sqrt(F.reshape(var, bc) + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "InstanceNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        eps = float(_attr(node, "epsilon", 1e-5))
        axes = tuple(range(2, len(x.shape)))
        nd = len(x.shape) - 2
        bc = [1, -1] + [1]*nd
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x-mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        return [F.reshape(scale, bc) * x_n + F.reshape(B_, bc)]

    if op == "LayerNormalization":
        x = get(0); scale = get(1); B_ = get(2)
        axis = int(_attr(node, "axis", -1))
        eps  = float(_attr(node, "epsilon", 1e-5))
        ndim = len(x.shape)
        norm_axis = axis % ndim
        axes = tuple(range(norm_axis, ndim))
        mean = F.mean(x, axis=axes, keepdims=True)
        var  = F.mean((x - mean)**2, axis=axes, keepdims=True)
        x_n  = (x - mean) / F.sqrt(var + np.float32(eps))
        if scale is not None: x_n = x_n * scale
        if B_ is not None:    x_n = x_n + B_
        return [x_n]

    # ── Pooling ──────────────────────────────────────────────────────────────
    if op in ("MaxPool", "AveragePool"):
        return _pool(node, get, F)

    if op == "GlobalAveragePool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.mean(x, axis=axes, keepdims=True)]

    if op == "GlobalMaxPool":
        x = get(0)
        axes = tuple(range(2, len(x.shape)))
        return [F.max(x, axis=axes, keepdims=True)]

    # ── Reductions ───────────────────────────────────────────────────────────
    if op == "ReduceMean":
        x = get(0)
        axes = _attr(node, "axes", None)
        if axes is None:
            at = get(1)
            if at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.mean(x, axis=ax, keepdims=kd)]

    if op == "ReduceSum":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sum(x, axis=ax, keepdims=kd)]

    if op == "ReduceMax":
        x = get(0)
        axes = _attr(node, "axes", None)
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.max(x, axis=ax, keepdims=kd)]

    if op == "ReduceL2":
        x = get(0)
        at = get(1); axes = _attr(node, "axes", None)
        if axes is None and at is not None: axes = _np(at).tolist()
        kd = bool(_attr(node, "keepdims", 1))
        ax = tuple(int(a) for a in axes) if axes else None
        return [F.sqrt(F.sum(x*x, axis=ax, keepdims=kd))]

    # ── Misc ─────────────────────────────────────────────────────────────────
    if op == "Where":
        return [F.where(get(0), get(1), get(2))]

    if op == "DepthToSpace":
        x = get(0)
        bs   = int(_attr(node, "blocksize", 2))
        mode = _attr(node, "mode", b"DCR")
        if isinstance(mode, bytes): mode = mode.decode()
        N, C, H, W = x.shape
        if mode == "DCR":
            x = F.reshape(x, [N, bs, bs, C//(bs*bs), H, W])
            x = F.transpose(x, [0, 3, 4, 1, 5, 2])
        else:  # CRD
            x = F.reshape(x, [N, C//(bs*bs), bs, bs, H, W])
            x = F.transpose(x, [0, 1, 4, 2, 5, 3])
        return [F.reshape(x, [N, C//(bs*bs), H*bs, W*bs])]

    if op == "Resize":
        return _resize(node, get, F)

    if op == "ConstantOfShape":
        shape = _np(get(0)).tolist()
        val_attr = _attr(node, "value", None)
        val = float(val_attr.flat[0]) if val_attr is not None else 0.0
        return [F.full(shape, np.float32(val)) if hasattr(F,'full')
                else np.full(shape, np.float32(val))]

    if op == "Shape":
        x = get(0)
        return [np.array(x.shape, dtype=np.int64)]

    if op == "Reciprocal":
        return [np.float32(1.0) / get(0)]

    if op in ("Equal", "Less", "Greater", "Not", "LessOrEqual", "GreaterOrEqual"):
        a, b = get(0), get(1)
        if op == "Equal":         return [a == b]
        if op == "Less":          return [a < b]
        if op == "Greater":       return [a > b]
        if op == "LessOrEqual":   return [a <= b]
        if op == "GreaterOrEqual":return [a >= b]
        if op == "Not":           return [~get(0)]

    if op == "CumSum":
        x = get(0)
        axis = int(_np(get(1)).flat[0])
        return [F.cumsum(x, axis=axis) if hasattr(F, 'cumsum') else F.cumulative_sum(x, axis=axis)]

    raise NotImplementedError(f"Unsupported ONNX op: {op}")


# ── Framework detection ───────────────────────────────────────────────────────

def _is_jax_module(F) -> bool:
    """Return True if F is jax.numpy (not tf.experimental.numpy)."""
    try:
        import jax.numpy as _jnp
        return F is _jnp
    except ImportError:
        return False


# ── Conv helper ──────────────────────────────────────────────────────────────

def _conv(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        dn = lax.conv_dimension_numbers(x.shape, w.shape, ("NCHW","OIHW","NCHW"))
        padding = ((int(pads[0]), int(pads[2])), (int(pads[1]), int(pads[3])))
        y = lax.conv_general_dilated(
            x, w,
            window_strides=[int(s) for s in strides],
            padding=padding,
            lhs_dilation=(1,1),
            rhs_dilation=[int(d) for d in dilations],
            dimension_numbers=dn,
            feature_group_count=group,
        )
    else:
        import tensorflow as tf
        # TF conv: NHWC format
        x_nhwc = tf.transpose(x, [0,2,3,1])
        w_hwio = tf.transpose(w, [2,3,1,0])  # OIHW → HWIO
        if group == 1:
            y_nhwc = tf.nn.conv2d(
                x_nhwc, w_hwio,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        else:
            # depthwise conv: w is [C,1,kH,kW] → need [kH,kW,C,1] for tf
            w_dwconv = tf.transpose(w, [2,3,0,1])  # [kH,kW,C,1]
            y_nhwc = tf.nn.depthwise_conv2d(
                x_nhwc, w_dwconv,
                strides=[1, int(strides[0]), int(strides[1]), 1],
                padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
                dilations=[int(dilations[0]), int(dilations[1])],
            )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── ConvTranspose helper ──────────────────────────────────────────────────────

def _conv_transpose(node, get, F):
    x = get(0); w = get(1); b = get(2)
    pads      = _attr(node, "pads",      [0,0,0,0])
    strides   = _attr(node, "strides",   [1,1])
    dilations = _attr(node, "dilations", [1,1])
    op_pads   = _attr(node, "output_padding", [0,0])
    group     = int(_attr(node, "group", 1))

    if _is_jax_module(F):
        import jax.lax as lax
        # ONNX ConvTranspose: w is [C_in, C_out/group, kH, kW].
        # Implement as dilated conv (lhs_dilation = strides) with spatially-flipped
        # and transposed weight → [C_out, C_in, kH, kW] in OIHW format.
        # Padding: for each spatial dim, pad = kernel - 1 - original_pad.
        kH = int(w.shape[2]); kW = int(w.shape[3])
        sH = int(strides[0]); sW = int(strides[1])
        dH = int(dilations[0]); dW = int(dilations[1])
        # Effective kernel size with dilation
        ekH = dH * (kH - 1) + 1; ekW = dW * (kW - 1) + 1
        # Transpose weight: [C_in, C_out, kH, kW] → [C_out, C_in, kH, kW], flip spatially
        w_t = F.transpose(w, (1, 0, 2, 3))[:, :, ::-1, ::-1]
        pad_h_top = ekH - 1 - int(pads[0]); pad_h_bot = ekH - 1 - int(pads[2]) + int(op_pads[0])
        pad_w_left = ekW - 1 - int(pads[1]); pad_w_right = ekW - 1 - int(pads[3]) + int(op_pads[1])
        y = lax.conv_general_dilated(
            x, w_t,
            window_strides=(1, 1),
            padding=((pad_h_top, pad_h_bot), (pad_w_left, pad_w_right)),
            lhs_dilation=(sH, sW),
            rhs_dilation=(dH, dW),
            feature_group_count=group,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'),
        )
    else:
        import tensorflow as tf
        # For ConvTranspose: w is [C_in, C_out/group, kH, kW] in ONNX
        # TF conv2d_transpose expects [kH, kW, C_out, C_in]
        x_nhwc = tf.transpose(x, [0,2,3,1])
        N, H_in, W_in, C_in = [int(d) for d in x_nhwc.shape]
        C_out = int(w.shape[1]) * group
        kH, kW = int(w.shape[2]), int(w.shape[3])
        sH, sW = int(strides[0]), int(strides[1])
        H_out = (H_in - 1) * sH - int(pads[0]) - int(pads[2]) + kH + int(op_pads[0])
        W_out = (W_in - 1) * sW - int(pads[1]) - int(pads[3]) + kW + int(op_pads[1])
        w_tf = tf.transpose(w, [2,3,1,0])  # [kH,kW,C_out/g,C_in]
        output_shape = [N, H_out, W_out, C_out]
        y_nhwc = tf.nn.conv2d_transpose(
            x_nhwc, w_tf,
            output_shape=output_shape,
            strides=[1, sH, sW, 1],
            padding=[[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]],
        )
        y = tf.transpose(y_nhwc, [0,3,1,2])

    if b is not None:
        y = y + F.reshape(b, [1,-1,1,1])
    return [y]


# ── Pool helper ──────────────────────────────────────────────────────────────

def _pool(node, get, F):
    op = node.op_type
    x = get(0)
    k         = _attr(node, "kernel_shape", [2,2])
    strides   = _attr(node, "strides",      [1,1])
    pads      = _attr(node, "pads",         [0,0,0,0])
    dilations = _attr(node, "dilations",    [1,1])
    ceil_mode = int(_attr(node, "ceil_mode", 0))

    dH, dW = int(dilations[0]), int(dilations[1])

    if _is_jax_module(F):
        import jax.lax as lax
        import jax.numpy as jnp
        pH0, pH1 = int(pads[0]), int(pads[2])
        pW0, pW1 = int(pads[1]), int(pads[3])
        if ceil_mode == 1:
            # Add extra right/bottom padding so lax.reduce_window matches ceil-mode output size
            in_H = int(x.shape[2]); in_W = int(x.shape[3])
            sH = int(strides[0]);   sW = int(strides[1])
            ekH = dH * (int(k[0]) - 1) + 1; ekW = dW * (int(k[1]) - 1) + 1
            rem_h = (in_H + pH0 + pH1 - ekH) % sH
            rem_w = (in_W + pW0 + pW1 - ekW) % sW
            pH1 += (sH - rem_h) if rem_h != 0 else 0
            pW1 += (sW - rem_w) if rem_w != 0 else 0
        pad_h = (pH0, pH1); pad_w = (pW0, pW1)
        padding = ((0,0),(0,0), pad_h, pad_w)
        window = (1, 1, int(k[0]), int(k[1]))
        str_   = (1, 1, int(strides[0]), int(strides[1]))
        win_dil = (1, 1, dH, dW)
        if op == "MaxPool":
            y = lax.reduce_window(x, -jnp.inf, lax.max, window, str_, padding,
                                  window_dilation=win_dil)
        else:
            ones = F.ones_like(x)
            s = lax.reduce_window(x,    0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            n = lax.reduce_window(ones, 0.0, lax.add, window, str_, padding,
                                  window_dilation=win_dil)
            y = s / n
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        ksize   = [1, int(k[0]),       int(k[1]),       1]
        str_tf  = [1, int(strides[0]), int(strides[1]), 1]
        paddings_tf = [[0,0],[int(pads[0]),int(pads[2])],[int(pads[1]),int(pads[3])],[0,0]]
        if op == "MaxPool" and (dH > 1 or dW > 1):
            # TF max_pool2d has no dilation support; use extract_patches + reduce_max
            kH, kW = int(k[0]), int(k[1])
            pH0, pH1 = int(pads[0]), int(pads[2])
            pW0, pW1 = int(pads[1]), int(pads[3])
            x_pad = tf.pad(x_nhwc, [[0,0],[pH0,pH1],[pW0,pW1],[0,0]],
                           constant_values=-1e9)
            patches = tf.image.extract_patches(
                x_pad,
                sizes=[1, kH, kW, 1],
                strides=str_tf,
                rates=[1, dH, dW, 1],
                padding="VALID",
            )
            N_, H_out, W_out, C_ = [int(d) for d in x_nhwc.shape]
            H_out2 = patches.shape[1]; W_out2 = patches.shape[2]
            C_in = int(x_nhwc.shape[-1])
            patches_r = tf.reshape(patches, [-1, H_out2, W_out2, kH * kW, C_in])
            y_nhwc = tf.reduce_max(patches_r, axis=3)
        elif op == "MaxPool":
            y_nhwc = tf.nn.max_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        else:
            # avg_pool doesn't support dilations in TF, treat as no dilation
            y_nhwc = tf.nn.avg_pool2d(x_nhwc, ksize, str_tf, padding=paddings_tf)
        y = tf.transpose(y_nhwc, [0,3,1,2])
    return [y]


# ── Resize helper ─────────────────────────────────────────────────────────────

def _resize(node, get, F):
    x = get(0)
    scales_t = get(2); sizes_t = get(3)
    mode = _attr(node, "mode", b"nearest")
    if isinstance(mode, bytes): mode = mode.decode()
    N, C = int(x.shape[0]), int(x.shape[1])
    if scales_t is not None:
        scales = _np(scales_t).tolist()
        H_new = int(int(x.shape[2]) * scales[2])
        W_new = int(int(x.shape[3]) * scales[3])
    else:
        ts = _np(sizes_t).tolist()
        H_new, W_new = int(ts[2]), int(ts[3])

    if _is_jax_module(F):
        import jax.image as ji
        import jax.numpy as jnp
        x_nhwc = jnp.transpose(x, (0,2,3,1))
        method = "nearest" if "nearest" in mode else "linear"
        y_nhwc = ji.resize(x_nhwc, (N, H_new, W_new, C), method=method)
        return [jnp.transpose(y_nhwc, (0,3,1,2))]
    else:
        import tensorflow as tf
        x_nhwc = tf.transpose(x, [0,2,3,1])
        method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if "nearest" in mode \
                 else tf.image.ResizeMethod.BILINEAR
        y_nhwc = tf.image.resize(x_nhwc, [H_new, W_new], method=method)
        return [tf.transpose(y_nhwc, [0,3,1,2])]
