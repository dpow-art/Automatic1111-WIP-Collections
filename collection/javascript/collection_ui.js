(function () {
  function setNumberInput(input, value) {
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function bindCollectionRoot() {
    const root = gradioApp().querySelector('#collection-root');
    if (!root || root.dataset.collectionBound === '1') return;
    root.dataset.collectionBound = '1';

    root.addEventListener('click', (event) => {
      const collectionButton = event.target.closest('.collection-row');
      if (collectionButton) {
        const id = collectionButton.dataset.collectionId;
        const inputs = gradioApp().querySelectorAll('#collection-main input[type="number"]');
        setNumberInput(inputs[0], id);
      }

      const cardButton = event.target.closest('.collection-card');
      if (cardButton) {
        const cardId = cardButton.dataset.itemId;
        const inputs = gradioApp().querySelectorAll('#collection-main input[type="number"]');
        setNumberInput(inputs[1], cardId);
      }
    });
  }

  onUiLoaded(() => {
    bindCollectionRoot();
  });
})();
